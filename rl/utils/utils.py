from sample_factory.utils.utils import log
from envs.doom.doom_utils import make_doom_env
from envs.doom.doom_params import add_doom_env_args, doom_override_defaults
from sample_factory.envs.env_registry import global_env_registry

def register_custom_components():
    global_env_registry().register_env(
      env_name_prefix='doomsound_',
      make_env_func=make_doom_env ,
      add_extra_params_func=add_doom_env_args,
      override_default_params_func=doom_override_defaults,
    )