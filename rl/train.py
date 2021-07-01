import sys

from envs.doom.doom_utils import make_doom_env
# from sample_factory.algorithms.utils.arguments import default_cfg
from rl.utils.utils import register_custom_components
from envs.doom.doom_params import add_doom_env_args
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.utils.utils import str2bool
from sample_factory.run_algorithm import run_algorithm

def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    register_custom_components()
    parser = arg_parser()
    cfg = parse_args(parser=parser)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())