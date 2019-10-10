from contextlib import contextmanager, ExitStack
import io
import os
import random
import re
import shutil
import subprocess
import tempfile


def temp_dir():
    # Should we instead set tempfile.tempdir ? How to reliably do that for
    # all executions? Need to investigate how that works with pyspark workers.
    if 'LOCAL_DIRS' in os.environ:
        dirs = os.environ['LOCAL_DIRS'].split(',')
        return random.choice(dirs)
    return None


@contextmanager
def as_output_file(path, mode='w', overwrite=False):
    if path[:7] == 'hdfs://':
        f = tempfile.NamedTemporaryFile(dir=temp_dir(), mode=mode)
        yield f
        f.flush()
        # Make the directory if it doesn't already exist
        subprocess.check_call(['hdfs', 'dfs', '-mkdir', '-p', os.path.dirname(path)])
        # Copy our local data into it
        put_cmd = ['hdfs', 'dfs', '-put']
        if overwrite:
            put_cmd.append('-f')
        put_cmd += [f.name, path]
        subprocess.check_call(put_cmd)
    else:
        if path[:len("file:/")] == "file:/":
            path = path[len("file:"):]
        with open(path, mode) as f:
            yield f


@contextmanager
def as_local_path(path):
    if path[0] == '/':
        yield path
    elif path[:len("file:/")] == "file:/":
        yield path[len("file:"):]
    else:
        with tempfile.NamedTemporaryFile(dir=temp_dir()) as local:
            os.unlink(local.name)
            subprocess.check_call(['hdfs', 'dfs', '-copyToLocal', path, local.name])
            yield local.name


@contextmanager
def as_local_paths(paths):
    with ExitStack() as stack:
        yield [stack.enter_context(as_local_path(path)) for path in paths]


@contextmanager
def as_output_files(paths):
    with ExitStack() as stack:
        yield [stack.enter_context(as_output_file(path)) for path in paths]


def hdfs_mkdir(path):
    # Will error if it already exists
    # TODO: Normalize error type?
    if path[:7] == 'hdfs://':
        subprocess.check_call(['hdfs', 'dfs', '-mkdir', path])
    else:
        os.mkdir(path)


def hdfs_unlink(*paths):
    remote = []
    for path in paths:
        if path[:7] == 'hdfs://':
            remote.append(path)
        else:
            if path[:len("file:/")] == "file:/":
                path = path[len("file:"):]
            os.unlink(path)
    if remote:
        subprocess.check_call(['hdfs', 'dfs', '-rm'] + remote)


def hdfs_rmdir(path):
    if path[:7] == 'hdfs://':
        subprocess.check_call(['hdfs', 'dfs', '-rm', '-r', '-f', path])
    else:
        shutil.rmtree(path)


@contextmanager
def hdfs_open_read(path):
    if path[:7] == 'hdfs://':
        content_bytes = subprocess.check_output(['hdfs', 'dfs', '-text', path])
        yield io.StringIO(content_bytes.decode('utf8'))
    else:
        with open(path, 'r') as f:
            yield f


def explode_ltr_model_definition(definition):
    """
    Parse a string describing a ltr featureset or model
    (featureset|model):name[@storename]

    Parameters
    ----------
    definition: string
        the model/featureset definition

    Returns
    -------
        list: 3 elements list: type, name, store
    """
    res = re.search('(featureset|model)+:([^@]+)(?:[@](.+))?$', definition)
    if res is None:
        raise ValueError("Cannot parse ltr model definition [%s]." % (definition))
    return res.groups()
