from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0, 1111, 2222, 3333, 4444]),
    # ('seed', [0, 1111, 2222]),
    ('env', ['doomsound_music_multi']),
    ('encoder_custom', ['vizdoom', 'vizdoomSoundSamples', 'vizdoomSoundLogMel', 'vizdoomSoundFFT']),
])

_experiments = [
    Experiment(
        'multi_sound_basic',
        # 'python -m algorithms.appo.train_appo --algo APPO --train_for_env_steps 100000000 --num_workers 18 --num_envs_per_worker 20 --experiment multisound_hell_vizdoomSoundFFT_3',
        # 'python -m algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        'python -m algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0',

        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_multi_sound_basic', experiments=_experiments)

# For brain run using:
# python -m sample_factory.runner.run --run=runner.doom_multi_sound_finder --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4