import sys

from envs.doom.doom_utils import make_doom_env
# from sample_factory.algorithms.utils.arguments import default_cfg
from rl.utils.utils import register_custom_components
from envs.doom.doom_params import add_doom_env_args
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.utils.utils import str2bool
from sample_factory.run_algorithm import run_algorithm

def main():
    register_custom_components()
    parser = arg_parser()
    cfg = parse_args(parser=parser)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())

# python -m rl.train --algo=APPO --env=doomsound_instruction --experiment=doom_instruction --encoder_custom=vizdoomSoundFFT --train_for_env_steps=500000000 --num_workers=24 --num_envs_per_worker=20 