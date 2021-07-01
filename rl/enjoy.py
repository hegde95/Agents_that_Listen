import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from rl.train import register_custom_components
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args


def main():
    """Script entry point."""
    register_custom_components()
    parser = arg_parser(evaluation = True)
    cfg = parse_args(parser=parser)
    status = enjoy(cfg)
    return status

# enjoy_quad
if __name__ == '__main__':
    sys.exit(main())