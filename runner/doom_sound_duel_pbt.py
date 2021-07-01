from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('ppo_epochs', [1]),
])

_experiment = Experiment(
    'duel_with_sound_self_play',
    # For Brain
    'python -m algorithms.appo.train_appo --env=doomsound_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=72 --num_envs_per_worker=16 --num_policies=4 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --encoder_custom=vizdoomSoundFFT --train_for_env_steps=2000000000',

    # Works locally
    # 'python -m algorithms.appo.train_appo --env=doom_duel_sound --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=18 --num_envs_per_worker=8 --num_policies=2 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=64 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --encoder_custom=vizdoomSoundFFT',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('duel_with_sound_self_play', experiments=[_experiment])

# For brain run using:
# python -m sample_factory.runner.run --run=runner.doom_sound_duel_pbt --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4