import pytest
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import types as T

import mjolnir.transform as mt


@pytest.mark.parametrize('expect,have,compatible,equal', [
    # Same column is compatible
    (T.StructType().add('id', 'integer'), T.StructType().add('id', 'integer'), True, True),
    # non-null is equal with null in both directions (sadly, hacked into place. This should be fixed at some point)
    (T.StructType().add('id', 'integer', True), T.StructType().add('id', 'integer', False), True, True),
    (T.StructType().add('id', 'integer', False), T.StructType().add('id', 'integer', True), True, True),
    # Extra columns in have are compatible but not equal
    (T.StructType(), T.StructType().add('id', 'integer'), True, False),
    # Extra columns in expect are not compatible
    (T.StructType().add('id', 'integer'), T.StructType(), False, False),
    # UDT's are equal in both directions
    (T.StructType().add('vec', VectorUDT()), T.StructType().add('vec', VectorUDT().sqlType()), True, True),
    (T.StructType().add('vec', VectorUDT().sqlType()), T.StructType().add('vec', VectorUDT()), True, True),
    # TODO: Test nested structures beyond implicitly through VectorUDT()
])
def test_schema_comparison(expect: T.StructType, have: T.StructType, compatible: bool, equal: bool) -> None:
    if equal and not compatible:
        raise Exception('Invalid constraint, can not be equal but not compatible')
    # functions return list of errors, not bool() returns true when everything is ok
    assert compatible is not bool(mt._verify_schema_compatability(expect, have))
    assert equal is not bool(mt._verify_schema_equality(expect, have))
