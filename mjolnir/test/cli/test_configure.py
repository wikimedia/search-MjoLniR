from argparse import ArgumentParser
import pytest

from mjolnir.cli import CLI_COMMANDS


@pytest.mark.parametrize('name,configure', CLI_COMMANDS.items())
def test_cli_args(fixture_factory, name, configure):
    parser = ArgumentParser()
    configure(parser)
    compare = fixture_factory('cli_args', name)
    compare(parser.format_help())
