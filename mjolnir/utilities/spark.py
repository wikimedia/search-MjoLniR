"""
Configuration based runner for spark commands

Running spark commands, and our own spark pipelines, have an
amazing number of arguments to setup. Delegate those arguments
into a config file that varies by command and by wiki.

The high level design here is to have a configuration file that
defines how to run various spark commands, and a process of templating
and configuration merging to define how to run those commands. The
basic structure of the configuration file is:

    working_dir: <path>
    global:
        <config>
        commands:
            bar:
                <config>
    profiles:
        foo:
            wikis: ["bazwiki", "bangwiki"]
            <config>
            commands:
                bar:
                    <config>

Configuration at the global level, profile level, and command level are
merged with precedence given to more specific items. Most notably this
means that global per-command <config> overrides top-level profile config.

The working_dir needs to be one such that the path to the python interpreter
inside the virtualenv has the same path on both the driver, where this code is
called from, and on the executors where a zip file of the virtualenv is
decompressed via sparks --archive foo.zip#path argument.

This final configuration is then used to define how to call spark. Most of
the complexity has been pushed to the configuration file where it's hopefully
easier to deal with. Only time will tell.
"""

from __future__ import absolute_import
import argparse
import datetime
import logging
import os
import pprint
import subprocess
import sys
import yaml


def dict_merge(base, override):
    """Recursively merge two dictionaries

    Parameters
    ----------
    base : dict
        Base dictionary for merge
    override : dict
        dictionary to override values from base with

    Returns
    -------
    dict
        Merged dictionary of base and overrides
    """
    # recursively merges dictionaries a and b, using b
    # as the overrides where applicable
    merged = base.copy()
    for k, v in override.items():
        if k in merged:
            if isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = dict_merge(merged[k], v)
            elif isinstance(merged[k], set) and isinstance(v, set):
                merged[k] = merged[k].union(v)
            else:
                merged[k] = v
        else:
            merged[k] = v
    return merged


def build_template_vars(template_vars, environment, marker):
    """Build the final template vars used in templating

    Parameters
    ----------
    template_vars : dict
        configuration specified to be the base template variables. These
        take preference over any other way to specify template variables.
    environment: dict
        configuration specified to be part of the environment. Merged
        into output, but with preference to template_vars argument
        from above. By merging we allows performing shell-like substitution
        of, for example, HOME.
    marker : str
        A unique marker for this run. Use isn't hard coded, but generally
        used in paths to have unique input/output directories.
    Returns
    -------
    dict
    """
    template_var_defaults = {
        'marker': marker,
    }

    # Merge environment into template vars giving precedent to template
    template_vars = dict_merge(environment, template_vars)
    # Merge in templating defaults
    template_vars = dict_merge(template_var_defaults, template_vars)
    # Everything must be strings to pass into subprocess.check_call:
    template_vars = {k: str(v) for k, v in template_vars.items()}

    # String replacement inside template_vars itself. This is a while
    # loop because a string may be replaced with another string that
    # contains unresolved templating vars.
    i = 0
    while any(['%(' in value for value in template_vars.values()]):
        template_vars = {k: v % template_vars for k, v in template_vars.items()}
        i += 1
        if i > 20:
            print(template_vars)
            raise Exception('Stuck in a loop')

    return template_vars


def apply_templating_recursive(data, template_vars):
    if isinstance(data, dict):
        return {k: apply_templating_recursive(v, template_vars) for k, v in data.items()}
    elif isinstance(data, list):
        return [apply_templating_recursive(v, template_vars) for v in data]
    elif isinstance(data, set):
        return set([apply_templating_recursive(v, template_vars) for v in data])
    else:
        try:
            return str(data) % template_vars
        except ValueError:
            raise ValueError('Failed formatting: %s' % (str(data)))


def load_config(stream, marker, template_overrides):
    """Load a configuration file and transform it into our final configuration

    Mostly applies a series of merges so command definitions override profile
    definitions which override global definitions. Then applies template variables
    in the form of %(foo)s using basic python string formatting.

    Parameters
    ----------
    stream : ???
        A file-like stream to read the configuration from
    marker : str
        A unique marker for this run. Use isn't hard coded, but generally
        used in paths to have unique input/output directories.
    template_overrides:
        Overrides applied to all templating variables. Generally these
        come from the command line.

    Returns
    -------
    global_profile : dict
        Top-level profile  containing all globally configured items
    profiles : dict
        Named profiles. The values in this dict are all the same shape
        as global_profile above but specialized to a particular set of
        wikis.
    """
    config = yaml.load(stream)

    if 'working_dir' in config:
        working_dir = config['working_dir']
    else:
        # TODO: This is wrong when running via the virtualenv?
        working_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    global_profile = config['global'].copy()
    # Merge in specific pieces of global environment, preferring actual environment over config
    global_profile['environment'] = dict_merge(global_profile.get('environment', {}), {
        'HOME': os.environ['HOME'],
        'USER': os.environ['USER'],
    })

    def build_profile(name, profile):
        # Apply profile overrides to global config
        profile = dict_merge(global_profile, profile)
        # Template vars should not propagate deeper into the stack.
        template_vars = dict_merge(profile.pop('template_vars', {}), template_overrides)
        # create an explicit profile name and working_dir var to reference
        for key, value in [('profile_name', name), ('working_dir', working_dir)]:
            if key in template_vars:
                raise Exception("%s defined externally" % (key))
            template_vars[key] = value
        # Take the commands and wikis. These will be returned, everything else will be merged
        # into the commands
        commands = profile.pop('commands', {})
        wikis = profile.pop('wikis', [])

        # Merge everything else(paths, environment, etc) into each command, with
        # the commmand taking precedence, and apply templating.
        for name, config in commands.items():
            merged = dict_merge(profile, config)
            command_template_vars = build_template_vars(
                dict_merge(template_vars, merged.pop('template_vars', {})),
                merged['environment'], marker)
            commands[name] = apply_templating_recursive(merged, command_template_vars)

        return {
            "wikis": wikis,
            "commands": commands,
        }

    profiles = {name: build_profile(name, profile) for name, profile in config['profiles'].items()}
    global_profile = build_profile('global', global_profile)

    return working_dir, global_profile, profiles


def validate_profiles(config):
    """Validate each profile has some minimum amount of information required"""
    for name, profile in config.items():
        # Assert a minimum viable level of per-group configuration
        # TODO: Would be nice if somehow we could also verify there aren't
        # extra unused properties that the user thinks will do something but don't
        validate_config_level(name + '.', profile, {
            'wikis': [],
            'commands': {
                'pyspark': ['spark_command'],
                'data_pipeline': ['spark_command', 'mjolnir_utility_path', 'mjolnir_utility'],
                'make_folds': {
                    'spark_command': [],
                    'mjolnir_utility_path': [],
                    'mjolnir_utility': [],
                    'spark_conf': [
                        'spark.yarn.executor.memoryOverhead',
                    ],
                    'spark_args': [],
                    'cmd_args': ['num-workers', 'num-folds']
                },
                'training_pipeline': {
                    # Using an empty array requires the key exists, even if it doesn't
                    # contain sub-properties.
                    'spark_command': [],
                    'mjolnir_utility_path': [],
                    'mjolnir_utility': [],
                    'spark_conf': [
                        'spark.dynamicAllocation.maxExecutors',
                        'spark.yarn.executor.memoryOverhead',
                        'spark.task.cpus'
                    ],
                    'spark_args': ['executor-memory', 'executor-cores'],
                    'cmd_args': ['cv-jobs', 'final-trees']
                }
            }
        })

        train = profile['commands']['training_pipeline']
        if train['spark_conf']['spark.task.cpus'] != train['spark_args']['executor-cores']:
            raise Exception('Expected spark.task.cpus to equal executor-cores in group %s' % (name))


def validate_config_level(prefix, config, requirements):
    """Recursively validate a dict has expected keys"""
    if type(requirements) is dict:
        for key, sub_requirements in requirements.items():
            if key not in config:
                raise Exception('Expected %s%s to exist in configuration file but have %s'
                                % (prefix, key, config.keys()))
            validate_config_level('%s%s.' % (prefix, key), config[key], sub_requirements)
    elif type(requirements) is list:
        for key in requirements:
            if key not in config:
                raise Exception('Expected %s%s to exist in configuration file but have %s'
                                % (prefix, key, config))
    else:
        raise Exception('Unexpected requirements type: %s' % (type(requirements)))


def check_defaults(profile, sub_commands):
    """Check list of paths that should/not exist

    Parameters
    ----------
    profile : dict
        A single profile (either global or named) created by load_config
    sub_commands : str
        A list of commands to verify correctness of within the profile

    Returns
    -------
    set of str
        Set of errors found in the profile
    """
    errors = set()

    def negate(func):
        return lambda x: not func(x)

    functions = {
        'dir_exist': (negate(os.path.isdir), 'Missing Directory: %s'),
        'dir_not_exist': (os.path.isdir, 'Directory should not exist: %s'),
        'file_exist': (negate(os.path.isfile), 'Missing File: %s'),
        'file_not_exist': (os.path.isfile, 'File should not exist: 5s'),
    }

    for k, v in functions.items():
        func, message = v
        for command in sub_commands:
            try:
                for path in profile['commands'][command]['paths'][k]:
                    if func(path):
                        errors.add(message % path)
            except KeyError:
                pass

    return errors


def pretty_print_cli(args, env):
    """Make a long command line barely readable when printed.

    Not suitable for copy/paste. Just look and understand.
    """
    args = list(args)
    output = []
    if env is not None:
        output.append('%s \\' % (' '.join(["%s=%s" % (k, v) for k, v in env.items()])))
    if sum([len(x) for x in args]) < 80:
        output.append(' '.join(args))
    else:
        output.append('\t%s' % (args.pop(0)))
        while args:
            line = [args.pop(0)]
            if args and line[0][:2] == '--':
                line.append(args.pop(0))
            else:
                while args and args[0][0] != '-' and sum([len(x) for x in line]) < 60:
                    line.append(args.pop(0))
            output[-1] += ' \\'
            output.append('\t%s' % (' '.join(line)))
    print('\n'.join(output))


def build_mjolnir_utility(config):
    """Buiild arguments for calling mjolnir-utility.py

    Parameters
    ----------
    config : dict
        Configuration of the individual command to run
    """
    args = [config['mjolnir_utility_path'], config['mjolnir_utility']]

    try:
        # while sorting is not strictly necessary, dicts have non-deterministic
        # sorts which make testing difficult
        for k, v in sorted(config['cmd_args'].items(), key=lambda x: x[0]):
            for item in (v if isinstance(v, (list, set)) else [v]):
                args.append('--' + k)
                args.append(str(item))
    except KeyError:
        pass

    return args


def build_spark_command(config):
    """Build a configuration based command line to run something in spark

    Puts together a command line for submitting spark jobs to the
    cluster. Injects spark_conf and spark_args from the profile
    for the specified command into the command line.

    Parameters
    ----------
    config : dict
        Command configuration to build spark arguments for
    """
    args = [config['spark_command']]
    try:
        for k, v in sorted(config['spark_conf'].items(), key=lambda x: x[0]):
            args.append('--conf')
            args.append('%s=%s' % (k, str(v)))
    except KeyError:
        pass

    try:
        for k, v in sorted(config['spark_args'].items(), key=lambda x: x[0]):
            args.append('--' + k)
            args.append(str(v))
    except KeyError:
        pass

    return args


def subprocess_check_call(args, env=None):
    """Helper function to only run commands if we are not using dry run"""
    print("Running Command:")
    pretty_print_cli(args, env=env)
    if DRY_RUN:
        return 0
    else:
        retval = subprocess.check_call(args, env=env)
        if retval is not 0:
            raise Exception("Subprocess returned non-zero exit code: %d" % (retval))


# Past here are the actual command definitions

def collect(global_profile, profiles):
    """Run mjolnir data pipeline"""
    all_wikis = [wiki for group in profiles.values() for wiki in group['wikis']]
    config = global_profile['commands']['data_pipeline']

    cmd = build_spark_command(config) + build_mjolnir_utility(config) + all_wikis
    subprocess_check_call(cmd, env=config['environment'])


def feature_selection(global_profile, profiles):
    """Run feature selection against collected data"""
    all_wikis = [wiki for group in profiles.values() for wiki in group['wikis']]
    config = global_profile['commands']['feature_selection']
    cmd = build_spark_command(config) + build_mjolnir_utility(config) + all_wikis
    subprocess_check_call(cmd, env=config['environment'])


def make_folds(global_profile, profiles):
    for name, profile in profiles.items():
        config = profile['commands']['make_folds']
        cmd = build_spark_command(config) + build_mjolnir_utility(config) + profile['wikis']
        subprocess_check_call(cmd, env=config['environment'])


def train(global_profile, profiles):
    """Run mjolnir training pipeline"""
    for name, profile in profiles.items():
        config = profile['commands']['training_pipeline']
        cmd = build_spark_command(config) + build_mjolnir_utility(config) + profile['wikis']
        subprocess_check_call(cmd, env=config['environment'])


def collect_and_train(global_profile, profiles):
    """Run data and training pipelines"""
    collect(global_profile, profiles)
    feature_selection(global_profile, profiles)
    make_folds(global_profile, profiles)
    train(global_profile, profiles)

    # Cleanup training data from hdfs
    # TODO when ready
    hdfs_training_data_path = global_profile['commands']['data_pipeline']['cmd_args']['output-dir']
    subprocess_check_call(['hdfs', 'dfs', '-rm', '-r', '-f', hdfs_training_data_path])


def shell(command, global_profile, profiles):
    """Start the pyspark shell"""
    config = global_profile['commands'][command]
    cmd = build_spark_command(config)
    subprocess_check_call(cmd, env=config['environment'])


class KeyValueAction(argparse.Action):
    "Allows specifying multiple k=v parameters on command line"
    def __init__(self, *args, **kwargs):
        kwargs['default'] = {}
        kwargs['type'] = KeyValueAction.check
        super(KeyValueAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Need to wrap in dict to get a copy so we don't change the provided default
        current = dict(getattr(namespace, self.dest, {}))
        k, v = values
        current[k] = v
        setattr(namespace, self.dest, current)

    @staticmethod
    def check(value):
        if '=' not in value:
            raise argparse.ArgumentTypeError('%s is not a k=v string' % (value))
        return tuple(value.split('=', 2))


COMMANDS = {
    'collect': {
        'func': collect,
        'needed': ['data_pipeline'],
    },
    'train': {
        'func': train,
        'needed': ['training_pipeline'],
    },
    'feature_selection': {
        'func': feature_selection,
        'needed': ['feature_selection'],
    },
    'make_folds': {
        'func': make_folds,
        'needed': ['make_folds'],
    },
    'collect_and_train': {
        'func': collect_and_train,
        'needed': ['data_pipeline', 'feature_selection', 'make_folds', 'training_pipeline'],
    },
    'shell': {
        'func': lambda x, y: shell('pyspark', x, y),
        'needed': []
    },
    'shell_train': {
        'func': lambda x, y: shell('pyspark_train', x, y),
        'needed': []
    }
}


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Run pre-configured spark-submit commands')
    parser.add_argument(
        '-c', '--config', dest='config', type=str, required=True,
        default='/etc/mjolnir/spark.yaml',
        help='Path to yaml configuration file.')
    parser.add_argument(
        '-t', '--template-var', dest='template_vars', action=KeyValueAction,
        metavar='var=override', help='Override template variables')
    parser.add_argument(
        '-m', '--marker', dest='marker', type=str,
        default=datetime.date.today().strftime("%Y%m%d"),
        help='Marker to tag training and output directories with. Defaults to ymd.')
    parser.add_argument(
        '-d', '--dry-run', dest='dry_run', action='store_true',
        help='Print the commands that would be run, dont run them')
    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        help='Print the command definition and exit')
    parser.set_defaults(dry_run=False)
    parser.add_argument(
        'command', metavar='command', type=str, choices=set(COMMANDS.keys()),
        help='Command to run: ' + ', '.join(COMMANDS.keys()))
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='*', default=[],
        help='Limit wikis to this list')
    return parser


def main(**args):
    # Easiest to just make this global instead of passing around
    global DRY_RUN
    DRY_RUN = args['dry_run']
    # Merge global config with training group config
    with open(args['config'], 'r') as f:
        working_dir, global_profile, profiles = load_config(f, args['marker'], args['template_vars'])

    # Filter to selected wikis
    if args['wikis']:
        include = set(wiki for wiki in args['wikis'] if wiki[0] != '-')
        exclude = set(wiki[1:] for wiki in args['wikis'] if wiki[0] == '-')
        if include and exclude:
            include = include.difference(exclude)
            exclude = None
        if include:
            for name, group in profiles.items():
                group['wikis'] = [wiki for wiki in group['wikis'] if wiki in include]
        elif exclude:
            for name, group in profiles.items():
                group['wikis'] = [wiki for wiki in group['wikis'] if wiki not in exclude]
    # Filter groups with no defined wikis
    profiles = {name: group for name, group in profiles.items() if group['wikis']}

    if args['debug']:
        print("Global Config: ")
        pprint.pprint(global_profile)
        print("\n\nProfiles:")
        pprint.pprint(profiles)
    else:
        command = COMMANDS[args['command']]

        # This is necessary because the path to the virtualenv needs to be the same
        # locally as it is on the remote executors. This places the venv at ./venv.
        # probably something less hard coded could be done but this works for now.
        # This needs to occur before validation so checks against PYSPARK_PYTHON
        # can verify the correct availability.
        # TODO: This may not be right, and isn't overridable ...
        os.chdir(working_dir)

        # Early-exit if there are configuration problems
        validate_profiles(profiles)
        errors = check_defaults(global_profile, command['needed'])
        for name, profile in profiles.items():
            errors = errors.union(check_defaults(profile, command['needed']))
        if errors:
            for error in errors:
                print(error)
            if not DRY_RUN:
                sys.exit(1)

        # Finally do the thing
        command['func'](global_profile, profiles)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
