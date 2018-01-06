from contextlib import contextmanager
from copy import deepcopy
import mjolnir.utils
import pytest


class FailTestException(Exception):
    pass


def make_multi_manager(pre_raise, post_raise, transitions):
    pre_raise = list(pre_raise)
    post_raise = list(post_raise)

    @contextmanager
    def f(data):
        if pre_raise.pop(0):
            transitions.append(('pre-raise', data))
            raise FailTestException('pre-raise: ' + str(data))
        transitions.append(('begin', data))
        yield data
        transitions.append(('end', data))
        if post_raise.pop(0):
            transitions.append(('post raise', data))
            raise FailTestException('post raise: ' + str(data))

    return mjolnir.utils.multi_with(f)


def generate_fixtures():
    tests = []
    n = 3

    success = [False] * n
    expect_success = [
        success,
        success,
        [('begin', i) for i in range(n)] +
        [('inside', None)] +
        [('end', i) for i in range(n)]]

    tests.append(expect_success)

    expect_failure = deepcopy(expect_success)
    expect_failure[0][1] = True
    expect_failure[2] = [('begin', 0), ('pre-raise', 1), ('end', 0)]
    tests.append(expect_failure)

    expect_failure = deepcopy(expect_success)
    expect_failure[0][n - 1] = True
    expect_failure[2] = [('begin', i) for i in range(n - 1)] + \
        [('pre-raise', n - 1)] + \
        [('end', i) for i in range(n-1)]
    tests.append(expect_failure)

    return ('pre_raise,post_raise,expected', tests)


@pytest.mark.parametrize(*generate_fixtures())
def test_multi_with_cleanup(pre_raise, post_raise, expected):
    n = 3
    transitions = []
    g = make_multi_manager(pre_raise, post_raise, transitions)
    try:
        with g(list(range(n))):
            transitions.append(('inside', None))
    except FailTestException:
        pass
    assert expected == transitions
