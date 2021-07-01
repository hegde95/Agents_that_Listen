from gym.spaces import Discrete

from envs.doom.action_space import doom_action_space, \
    doom_action_space_full_discretized, doom_action_space_basic, doom_action_space_discretized_no_weap, \
    doom_action_space_extended, doom_turn_and_attack_only
from envs.doom.doom_gym import VizdoomEnv

from envs.doom.doom_model import register_models
from envs.doom.wrappers.additional_input import DoomAdditionalInput
from envs.doom.wrappers.bot_difficulty import BotDifficultyWrapper
from envs.doom.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from envs.doom.wrappers.observation_space import SetResolutionWrapper, resolutions
from envs.doom.wrappers.reward_shaping import true_reward_final_position, DoomRewardShapingWrapper, \
    REWARD_SHAPING_DEATHMATCH_V0, true_reward_frags, REWARD_SHAPING_DEATHMATCH_V1, REWARD_SHAPING_BATTLE
from envs.doom.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping
from sample_factory.envs.env_wrappers import ResizeWrapper, RewardScalingWrapper, TimeLimitWrapper, RecordingWrapper, \
    PixelFormatChwWrapper
from envs.doom.wrappers.sound_wrapper import DoomSound
from sample_factory.utils.utils import log


VIZDOOM_INITIALIZED = False


class DoomSpec:
    def __init__(
            self, name, env_spec_file, action_space, reward_scaling=1.0, default_timeout=-1,
            num_agents=1, num_bots=0,
            respawn_delay=0, timelimit=4.0,
            extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        # CLI arguments override this (see enjoy_rllib.py)
        self.num_bots = num_bots

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers


SOUND_INPUT = (DoomSound, {}) # sound input
ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
BATTLE_REWARD_SHAPING = (DoomRewardShapingWrapper, dict(reward_shaping_scheme=REWARD_SHAPING_BATTLE, true_reward_func=None))  # "true" reward None means just the env reward (monster kills)
BOTS_REWARD_SHAPING = (DoomRewardShapingWrapper, dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V0, true_reward_func=true_reward_frags))
DEATHMATCH_REWARD_SHAPING = (DoomRewardShapingWrapper, dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V1, true_reward_func=true_reward_final_position))


DOOM_ENVS = [
    DoomSpec('doomsound_music_multi', 'music_sound_multi.cfg', doom_action_space_basic(), 1.0),

    DoomSpec('doomsound_music_single', 'music_sound.cfg', doom_action_space_basic(), 1.0),

    DoomSpec('doomsound_instruction', 'instruction_sound.cfg', doom_action_space_basic(), 1.0),
    DoomSpec('doomsound_once_instruction', 'instruction_once_sound.cfg', doom_action_space_basic(), 1.0),
    DoomSpec('doomsound_memory', 'memory_sound_finder.cfg', doom_action_space_basic(), 1.0),

    DoomSpec(
        'doomsound_duel',
        'ssl2.cfg',
        doom_action_space_full_discretized(with_use=True),
        1.0, int(1e9),
        num_agents=2, num_bots=0, respawn_delay=2,
        extra_wrappers=[ADDITIONAL_INPUT, DEATHMATCH_REWARD_SHAPING],
    ),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


# noinspection PyUnusedLocal
def make_doom_env_impl(
        doom_spec,
        cfg=None,
        env_config=None,
        skip_frames=None,
        episode_horizon=None,
        player_id=None, num_agents=None, max_num_players=None, num_bots=0,  # for multi-agent
        custom_resolution=None,
        **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if 'fps' in cfg else None
    async_mode = fps == 0

    if player_id is None:
        env = VizdoomEnv(
            doom_spec.action_space, doom_spec.env_spec_file, skip_frames=skip_frames, async_mode=async_mode,
            sound_enabled=cfg.doom_sound_enabled,
            sound_sampling_rate=cfg.sampling_rate,
            sound_obs_num_frames=cfg.num_frames,
        )
    else:
        timelimit = cfg.timelimit if cfg.timelimit is not None else doom_spec.timelimit

        from envs.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer
        env = VizdoomEnvMultiplayer(
            doom_spec.action_space, doom_spec.env_spec_file,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
            respawn_delay=doom_spec.respawn_delay,
            timelimit=timelimit,
            sound_enabled=cfg.doom_sound_enabled,
            sound_sampling_rate=cfg.sampling_rate,
            sound_obs_num_frames=cfg.num_frames,
        )

    record_to = cfg.record_to if 'record_to' in cfg else None
    should_record = False
    if env_config is None:
        should_record = True
    elif env_config.worker_index == 0 and env_config.vector_index == 0 and (player_id is None or player_id == 0):
        should_record = True

    if record_to is not None and should_record:
        env = RecordingWrapper(env, record_to, player_id)

    env = MultiplayerStatsWrapper(env)

    if num_bots > 0:
        bot_difficulty = cfg.start_bot_difficulty if 'start_bot_difficulty' in cfg else None
        env = BotDifficultyWrapper(env, bot_difficulty)

    resolution = custom_resolution
    if resolution is None:
        resolution = '256x144' if cfg.wide_aspect_ratio else '160x120'

    assert resolution in resolutions
    env = SetResolutionWrapper(env, resolution)  # default (wide aspect ratio)

    h, w, channels = env.observation_space.shape
    if w != cfg.res_w or h != cfg.res_h:
        env = ResizeWrapper(env, cfg.res_w, cfg.res_h, grayscale=False)

    log.info('Doom resolution: %s, resize resolution: %r', resolution, (cfg.res_w, cfg.res_h))

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    if timeout > 0:
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if 'pixel_format' in cfg else 'HWC'
    if pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if cfg.doom_sound_enabled:
        log.debug('Sound enabled! Adding additional sound observations')
        env = DoomSound(env)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    return env


def make_doom_multiplayer_env(doom_spec, cfg=None, env_config=None, **kwargs):
    skip_frames = cfg.env_frameskip

    if cfg.num_bots < 0:
        num_bots = doom_spec.num_bots
    else:
        num_bots = cfg.num_bots

    num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
    max_num_players = num_agents + cfg.num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_spec,
            cfg=cfg,
            player_id=player_id, num_agents=num_agents, max_num_players=max_num_players, num_bots=num_bots,
            skip_frames=1 if is_multiagent else skip_frames,  # multi-agent skipped frames are handled by the wrapper
            env_config=env_config,
            **kwargs,
        )

    if is_multiagent:
        # create a wrapper that treats multiple game instances as a single multi-agent environment

        from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv
        env = MultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
        )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        from envs.doom.multiplayer.doom_multiagent_wrapper import init_multiplayer_env
        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    return env


def make_doom_env(env_name, **kwargs):
    ensure_initialized()

    spec = doom_env_by_name(env_name)

    if spec.num_agents > 1 or spec.num_bots > 0:
        # requires multiplayer setup (e.g. at least a host, not a singleplayer game)
        return make_doom_multiplayer_env(spec, **kwargs)
    else:
        return make_doom_env_impl(spec, **kwargs)


def ensure_initialized():
    global VIZDOOM_INITIALIZED
    if VIZDOOM_INITIALIZED:
        return

    register_models()

    VIZDOOM_INITIALIZED = True
