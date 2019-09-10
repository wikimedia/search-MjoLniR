"""Helpers to construct a transformation pipeline

A transformer is, broadly speaking, a function that takes a DataFrame as
it's only argument and returns a new DataFrame. Using this fairly simple
definition we can compose together multiple transformations into new
transformations, and we can parameterize the transformations by defining
functions that take the parameters and return a transformer.

One downside of this approach is that it can be hard to track what is actually
in the DataFrame. A decorator, @typed_transformer, is provided to decorate
transformers with assertions that the schema matches an expected schema. This
helps to document what goes in and out of a transformation.
"""
import decimal
import functools
import os
from typing import cast, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from pyspark import RDD
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import Column, DataFrame, functions as F, SparkSession, types as T


_T = TypeVar('_T')
Transformer = Callable[[DataFrame], DataFrame]


# Read/Write individual partitions with schema compatability checks


def _partition_spec_ql(partition_spec: Mapping[str, str]) -> str:
    """Partition specification in HQL"""
    # TODO: don't quote numbers like refinery?
    return ','.join('{}="{}"'.format(k, v) for k, v in partition_spec.items())


def _add_partition_ql(
    table_name: str, partition_path: str, partition_spec: Mapping[str, str]
) -> str:
    """HQL statement to add partition to hive metastore"""
    hql_part = _partition_spec_ql(partition_spec)
    fmt = "ALTER TABLE {} ADD IF NOT EXISTS PARTITION ({}) LOCATION '{}'"
    return fmt.format(table_name, hql_part, partition_path)


def _describe_partition_ql(table_name: str, partition_spec: Mapping[str, str]) -> str:
    """HQL statement to describe a single partition.

    A useful side effect of this query is that running it will produce
    exceptions for partition not found and extra/missing partition columns.
    """
    hql_part = _partition_spec_ql(partition_spec)
    fmt = "DESCRIBE {} PARTITION ({})"
    return fmt.format(table_name, hql_part)


def _simplify_data_type(data_type: T.DataType) -> Tuple:
    """Simplify datatype into a tuple of equality information we care about

    Most notably this ignores nullability concerns due to hive not
    being able to represent not null in it's schemas.
    """
    try:
        # Normalize UDT into it's sql form. Allows comparison of schemas
        # from hive and spark.
        sql_type = data_type.sqlType()  # type: ignore
    except AttributeError:
        sql_type = data_type

    if isinstance(sql_type, T.StructType):
        return ('StructType', [(field.name, _simplify_data_type(field.dataType)) for field in sql_type])
    elif isinstance(sql_type, T.ArrayType):
        return ('ArrayType', _simplify_data_type(sql_type.elementType))
    else:
        return (type(sql_type).__name__,)


def _verify_schema_compatability(expect: T.StructType, have: T.StructType) -> List[str]:
    """Verify all expected fields and types are present

    Allows additional columns in the `have` schema. Additionally
    allows relaxing nullability """
    errors = []
    for expect_field in expect:
        try:
            have_field = have[expect_field.name]
        except KeyError:
            errors.append('Field {} missing. Have: {}'.format(expect_field.name, ','.join(have.names)))
            continue
        expect_type = _simplify_data_type(expect_field.dataType)
        have_type = _simplify_data_type(have_field.dataType)
        if expect_type != have_type:
            errors.append('Field {} has incompatible data types: expect {} != have {}'.format(
                          expect_field.name, expect_type, have_type))
    return errors


def _verify_schema_equality(expect: T.StructType, have: T.StructType) -> List[str]:
    """Verify the dataframe and table have equal schemas"""
    def resolve(schema, field_name) -> Optional[Tuple]:
        try:
            field = schema[field_name]
        except KeyError:
            return None
        return _simplify_data_type(field.dataType)

    errors = []
    for field_name in set(expect.names).union(have.names):
        expect_type = resolve(expect, field_name)
        if expect_type is None:
            errors.append('Extra field in provided schema: {}'.format(field_name))
            continue

        have_type = resolve(have, field_name)
        if have_type is None:
            errors.append('Missing field in provided schema: {}'.format(field_name))
            continue

        if expect_type != have_type:
            fmt = 'Column {} of type {} does not match expected {}'
            errors.append(fmt.format(field_name, have_type, expect_type))
            continue
        # TODO: Test nullability? But hive doesn't track nullability, everything is nullable.
    return errors


def read_partition(
    spark: SparkSession,
    table: str,
    partition_spec: Mapping[str, str],
    schema: Optional[T.StructType] = None,
    direct_parquet_read: bool = False
) -> DataFrame:
    """Read a single partition from a hive table.

    Verifies the partition specification describes a complete partition,
    that the partition exists, and optionally that the table is compatible
    with an expected schema. The partition could still be empty.
    """
    # We don't need to do anything with the result, our goal is to
    # trigger AnalysisException when the arguments are invalid.
    spark.sql(_describe_partition_ql(table, partition_spec)).collect()

    partition_cond = F.lit(True)
    for k, v in partition_spec.items():
        partition_cond &= F.col(k) == v
    df = spark.read.table(table).where(partition_cond)
    # The df we have now has types defined by the hive table, but this downgrades
    # non-standard types like VectorUDT() to it's sql equivalent. Use the first
    # df to find the files, then read them directly.
    if direct_parquet_read:
        input_files = list(df._jdf.inputFiles())  # type: ignore
        input_dirs = set(os.path.dirname(path) for path in input_files)
        if len(input_dirs) != 1:
            raise Exception('Expected single directory containing partition data: [{}]'.format(
                '],['.join(input_files)))
        df = spark.read.parquet(list(input_dirs)[0])
    if schema is not None:
        # TODO: This only allows extra top level columns, anything
        # nested must be exactly the same. Fine for now.
        _verify_schema_compatability(schema, df.schema)
        df = df.select(*(field.name for field in schema))
    # Drop partitioning columns. These are not part of the mjolnir transformations, and
    # are only an implementation detail of putting them on disk and tracking history.
    return df.drop(*partition_spec.keys())


def write_partition(
    df: DataFrame, output_table: str, output_path: str,
    partition_spec: Mapping[str, str], mode: str = 'overwrite'
) -> None:
    """Write dataframe to disk as parquet and add to hive metastore"""
    for k, v in partition_spec.items():
        df = df.withColumn(k, F.lit(v))

    expect_schema = df.sql_ctx.read.table(output_table).schema
    errors = _verify_schema_equality(expect_schema, df.schema)
    if errors:
        raise Exception('Output table has incompatible schema: {}'.format(
            ', '.join(errors)))
    df.write.mode(mode).parquet(output_path)
    df.sql_ctx.sparkSession.sql(_add_partition_ql(
        output_table, output_path, partition_spec)).collect()


# Generic helpers for composing transformations


def par_transform(transformations: List[Transformer], mapper=map) -> Transformer:
    """Run a list of transformers as parallel transformations"""
    def transform(df: DataFrame) -> DataFrame:
        dfs = mapper(lambda fn: fn(df), transformations)
        return functools.reduce(DataFrame.union, dfs)
    return transform


def seq_transform(transformations: List[Transformer]) -> Transformer:
    """Run a list of transformers as a sequential transformation"""
    def transform(df: DataFrame) -> DataFrame:
        for fn in transformations:
            df = fn(df)
        return df
    return transform


# Generic type variable limited to values accepted by
# pyspark.sql.Column.__eq__ (and friends).
_LT = TypeVar('_LT', Column, bool, int, float, str, decimal.Decimal)


def for_each_item(
    col_name: str,
    items: List[_LT],
    transformer_factory: Callable[[_LT], Transformer],
    mapper=map
) -> Transformer:
    """Run a transformation for each value in a list of values"""
    # A lambda inside the list comprehension would capture `item`
    # by name, use a proper function to ensure item is captured
    # from a unique context.
    def restrict_to_item(item: _LT) -> Transformer:
        return lambda df: df.where(F.col(col_name) == item)

    transformers = [seq_transform([
        restrict_to_item(item),
        transformer_factory(item)
    ]) for item in items]

    return par_transform(transformers, mapper)


# Shared transformations


def identity(x: _T) -> _T:
    """Return argument as-is"""
    return x


def restrict_wikis(wikis: Optional[List[str]]) -> Transformer:
    """Optionally limit dataframe to specific set of wikis"""
    def transform(df: DataFrame) -> DataFrame:
        return df.where(F.col('wikiid').isin(wikis))

    if wikis is None:
        return identity
    elif len(wikis) == 0:
        raise Exception('No wikis provided')
    else:
        return transform


def assert_not_empty(df: DataFrame) -> DataFrame:
    """Fail the job if dataframe is empty.

    Spark will have to compute up to this point to determine if the df is
    empty, and will probably recompute all that later when the actual job is
    run.
    """
    if len(df.take(1)) == 0:
        raise Exception("DataFrame is empty")
    return df


def eagerly_cache(df: DataFrame) -> DataFrame:
    """Cache a dataframe in memory and pre-compute it before continuing"""
    df.cache()
    df.count()
    return df


def cache_to_disk(temp_dir: str, partition_by: str) -> Transformer:
    """Write a dataframe to disk partitioned by a column.

    Writes out the source dataframe partitioned by the provided
    column. The intention is for downstream tasks to construct
    a dataframe per partitioned value. When doing so this allows
    the downstream data frames to read individual columns for specific
    wikis from disk directly.

    Cleaning up the temp_dir is the callers responsibility and must
    be done after all transformations have executed to completion,
    likely after closing the SparkContext.

    TODO: This emits the same number of partitions for each partition col,
    while some may need 1 partition and others 1000. We would need count
    estimates to do that partitioning though.
    """
    def transform(df: DataFrame) -> DataFrame:
        df.write.partitionBy(partition_by).parquet(temp_dir)
        return df.sql_ctx.read.parquet(temp_dir)
    return transform


def temp_rename_col(
    orig_name: str, temp_name: str, fn: Transformer
) -> Transformer:
    """Rename a column within the context of the nested transformer"""
    def transform(df: DataFrame) -> DataFrame:
        return fn(df.withColumnRenamed(orig_name, temp_name)) \
            .withColumnRenamed(temp_name, orig_name)
    return transform


def partition_per_row(rdd: RDD) -> RDD:
    """Place each row in an RDD into a separate partition.

    Only useful if that row represents something large to be computed over,
    perhaps an external resource such as a multi-gb training dataset. The spark
    part of the dataset is expected to be tiny and easily fit in a single
    partition.
    """
    num_rows = rdd.count()
    # Help out mypy. Also don't use `identity`, as it somehow fails serialization
    partition_fn = cast(Callable[[int], int], lambda x: x)

    return (
        # bring everything together and assign each row a partition id
        rdd.repartition(1)
        .mapPartitions(lambda rows: enumerate(rows))
        # Partition by the new parition_id
        .partitionBy(num_rows, partition_fn)
        # Drop the partition id, giving back the origional shape
        .map(lambda pair: pair[1])
    )


# Shared joins


def _join(how, on):
    def fn(df_other: DataFrame) -> Transformer:
        def transform(df: DataFrame) -> DataFrame:
            return df.join(df_other, how=how, on=on)
        return transform
    return fn


join_cluster_by_cluster_id = _join('inner', ['wikiid', 'cluster_id'])
join_cluster_by_query = _join('inner', ['wikiid', 'query'])
join_labels = _join('inner', ['wikiid', 'query', 'page_id'])


# Helpers to document and verify input/output schemas

def check_schema(df: DataFrame, expect: T.StructType, context: Optional[str] = None) -> None:
    errors = _verify_schema_compatability(expect, df.schema)
    if errors:
        if context is None:
            context_str = ''
        else:
            context_str = context + '\n'
        raise Exception(context_str + '\n'.join(errors))


def typed_transformer(
    schema_in: Optional[T.StructType] = None,
    schema_out: Optional[T.StructType] = None,
    context: Optional[str] = None
) -> Callable[[Callable[..., Transformer]], Callable[..., Transformer]]:
    """Decorates a transformer factory with schema validation

    An idiom in transform is calling a function to return a Transform. This
    decorator can be applied to those factory functions to return transformers
    that apply runtime schema validation.
    """
    def decorate(fn: Callable[..., Transformer]) -> Callable[..., Transformer]:
        def error_context(kind: str) -> str:
            return 'While checking {} {}:'.format(fn.__name__ if context is None else context, kind)

        @functools.wraps(fn)
        def factory(*args, **kwargs) -> Transformer:
            transformer = fn(*args, **kwargs)

            @functools.wraps(transformer)
            def transform(df_in: DataFrame) -> DataFrame:
                if schema_in is not None:
                    check_schema(df_in, schema_in, error_context('schema_in'))
                    df_in = df_in.select(*schema_in.names)
                df_out = transformer(df_in)
                if schema_out is not None:
                    check_schema(df_out, schema_out, error_context('schema_out'))
                    df_out = df_out.select(*schema_out.names)
                return df_out
            return transform
        return factory
    return decorate


# Shared schemas between the primary mjolnir transformations. Transformations
# may require a schema with slightly more columns than they require to keep
# the total number of schemas low.


def _merge_schemas(*schemas: T.StructType):
    """Merge one or more spark schemas into a new schema"""
    fields = cast(Dict[str, T.StructField], {})
    errors = []
    for schema in schemas:
        for field in schema:
            if field.name not in fields:
                fields[field.name] = field
            elif field != fields[field.name]:
                errors.append('Incompatible fields: {} != {}'.format(field, fields[field.name]))
    if errors:
        raise Exception('\n'.join(errors))
    return T.StructType(list(fields.values()))


# Primary input schema from which most everything else is derived
QueryClicks = T.StructType([
    # wikiid and project represent same data in two ways
    T.StructField('wikiid', T.StringType(), nullable=False),
    T.StructField('project', T.StringType(), nullable=False),
    # Short term (10's of minutes) session token
    T.StructField('session_id', T.StringType(), nullable=False),
    T.StructField('query', T.StringType(), nullable=False),
    T.StructField('timestamp', T.LongType(), nullable=False),
    # All pages returned by query in order
    T.StructField('hit_page_ids', T.ArrayType(T.IntegerType()), nullable=False),
    # All pages clicked by user. This does not track the order of operations,
    # requiring assumptions about the order of clicks.
    T.StructField('click_page_ids', T.ArrayType(T.IntegerType()), nullable=False),
    # Debug info to associate with external data sources
    T.StructField('request_set_token', T.StringType(), nullable=False),
])


# Queries with same cluster_id are the same in some way.
# Each (wikiid, query) pair is distinct, a query must not have
# multiple cluster_id's.
QueryClustering = T.StructType([
    T.StructField('wikiid', T.StringType(), nullable=False),
    T.StructField('query', T.StringType(), nullable=False),
    T.StructField('cluster_id', T.LongType(), nullable=False),
])


# Only used as input, not output. Specifies a set of queries
# and related pages we are interested in operating on.
QueryPage = T.StructType([
    T.StructField('wikiid', T.StringType(), nullable=False),
    T.StructField('query', T.StringType(), nullable=False),
    T.StructField('page_id', T.IntegerType(), nullable=False),
])

# Various features describing the relationship between a query and page.
# Each QueryPage must be unique.
FeatureVectors = _merge_schemas(QueryPage) \
    .add(T.StructField('features', VectorUDT(), nullable=False))

# A QueryPage with some measure of relevance between the query and page.
# Each QueryPage must be unique.
LabeledQueryPage = _merge_schemas(QueryPage, QueryClustering) \
    .add(T.StructField('label', T.DoubleType(), nullable=False))


# Metadata about files formatted for training and stored in hadoop.
TrainingFiles = T.StructType([
    T.StructField('wikiid', T.StringType(), nullable=False),
    T.StructField('vec_format', T.StringType(), nullable=False),
    T.StructField('split_name', T.StringType(), nullable=False),
    T.StructField('path', T.StringType(), nullable=False),
    T.StructField('fold_id', T.IntegerType(), nullable=False),
])


ModelParameters = T.StructType([
    T.StructField('run_id', T.StringType(), nullable=False),
    T.StructField('parent_run_id', T.StringType(), nullable=True),
    T.StructField('wikiid', T.StringType(), nullable=False),
    T.StructField('started_at', T.TimestampType(), nullable=False),
    T.StructField('completed_at', T.TimestampType(), nullable=False),
    T.StructField('algorithm', T.StringType(), nullable=False),
    T.StructField('objective', T.StringType(), nullable=False),
    T.StructField('loss', T.DoubleType(), nullable=False),
    T.StructField('params', T.MapType(T.StringType(), T.StringType(), False), nullable=False),
    T.StructField('folds', T.ArrayType(TrainingFiles), nullable=False),
    T.StructField('metrics', T.ArrayType(T.StructType([
        T.StructField('key', T.StringType(), nullable=False),
        T.StructField('value', T.DoubleType(), nullable=False),
        T.StructField('step', T.IntegerType(), nullable=False),
        T.StructField('fold_id', T.IntegerType(), nullable=False),
        T.StructField('split', T.StringType(), nullable=False),
    ]))),
    T.StructField('artifacts', T.MapType(T.StringType(), T.StringType(), False), nullable=False),
])
# Custom hack that should probably move upstream into query generation


class ContentIndices:
    """Select content indices when collecting from kafka+es

    Mimic's the python dict interface used by our es query
    generation to convert wikiid into an elastic index name.
    Returns <wiki>_content for all possible wikis.
    """
    def __contains__(self, wikiid):
        return True

    def __getitem__(self, wikiid):
        return '{}_content'.format(wikiid)
