import datetime
import math
import os
import sys
import time
from collections import deque
from os.path import join

import numpy as np
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint, arg_parser
from rl.train import register_custom_components
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from envs.doom.doom_utils import make_doom_env


import cv2
from scipy.io.wavfile import write
import moviepy.editor as mpe
import matplotlib.pyplot as plot
from PIL import Image

# from envs.doom.doom_utils import make_doom_env
# from algorithms.utils.arguments import default_cfg

# def default_doom_cfg():
#     return default_cfg(env='doom_env')


# python -m demo.enjoy_appo --env doom_music_sound_multi --train_dir train_dir_for_demo --experiment 19_multi_sound_basic_see_4444_env_doom_music_sound_multi_enc_vizdoomSoundFFT --algo APPO --experiments_root ./
# python -m demo.enjoy_appo --env doom_sound_instruction --train_dir train_dir_for_demo --experiment 17_doom_sound_instruction_see_4444_env_doom_sound_instruction_enc_vizdoomSoundFFT --algo APPO --experiments_root ./
# python -m demo.enjoy_appo --env doom_once_sound_instruction --train_dir train_dir_for_demo --experiment 19_doom_sound_instruction_see_4444_env_doom_once_sound_instruction_enc_vizdoomSoundFFT --algo APPO --experiments_root ./
# python -m demo.enjoy_appo --env doom_memory_sound --train_dir train_dir_for_demo --experiment 01_doom_sound_memory_see_0_env_doom_memory_sound_enc_vizdoomSoundFFT --algo APPO --experiments_root ./

def enjoy(cfg, max_num_episodes=10, max_num_frames=1e9):
    # cfg.train_dir = "/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo"
    train_dir = cfg.train_dir
    cfg = load_from_checkpoint(cfg)
    cfg.train_dir = train_dir

    # cfg.train_dir = "/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo"
    # cfg.experiments_root = "./"
    cfg.record_to = False

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    # render_action_repeat = 1
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1
    # cfg.wide_aspect_ratio = True

    if cfg.record_to:
        tstamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        cfg.record_to = join(cfg.record_to, f'{cfg.experiment}', tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_doom_env(cfg.env, cfg=cfg, env_config=AttrDict({'worker_index': 0, 'vector_index': 0}), custom_resolution = '1920x1080')

    # env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    # env = create_env(cfg.env, cfg=cfg, env_config=AttrDict({'worker_index': 0, 'vector_index': 0}), custom_resolution = '256x144')
    env.seed(0)

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False


    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    print(checkpoints)
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = []
    audios = []
    screens = []
    true_rewards = deque([], maxlen=100)
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()
    # obs = [obs]

    with torch.no_grad():
        for _ in range(max_num_episodes):
            done = [False] * len(obs)
            rnn_states = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)

            episode_reward = 0

            while True:
                obs_torch = AttrDict(transform_dict_observations(obs))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()

                policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

                # sample actions from the distribution by default
                actions = policy_outputs.actions

                action_distribution = policy_outputs.action_distribution
                if isinstance(action_distribution, ContinuousActionDistribution):
                    if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                        actions = action_distribution.means

                actions = actions.cpu().numpy()

                rnn_states = policy_outputs.rnn_states

                # screen = obs[0]["obs"]
                # screen = env.unwrapped.state.screen_buffer
                audio = env.unwrapped.state.audio_buffer
                if audio is not None:
                    # scrn = np.swapaxes(np.swapaxes(screen,0,1),1,2)
                    # screens.append(scrn)
                    # img = Image.fromarray(scrn, 'RGB')
                    # img.show()
                    # screens.append(screen)
                    list_audio = list(audio)
                    # audios.extend(list_audio[:len(list_audio)])
                    audios.extend(list_audio)

                for _ in range(render_action_repeat):
                    if not cfg.no_render:
                        target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                        current_delay = time.time() - last_render_start
                        time_wait = target_delay - current_delay

                        if time_wait > 0:
                            # log.info('Wait time %.3f', time_wait)
                            time.sleep(time_wait)

                        last_render_start = time.time()
                        env.render()

                    obs, rew, done, infos = env.step(actions)
                    # obs = [obs]

                    if audio is not None:
                        screen = env.unwrapped.state.screen_buffer
                        scrn = np.swapaxes(np.swapaxes(screen,0,1),1,2)
                        screens.append(scrn)

                    episode_reward += np.mean(rew)
                    num_frames += 1

                    if all(done):
                        true_rewards.append(infos[0].get('true_reward', math.nan))
                        log.info('Episode finished at %d frames', num_frames)
                        if not math.isnan(np.mean(true_rewards)):
                            log.info('true rew %.3f avg true rew %.3f', true_rewards[-1], np.mean(true_rewards))

                        # VizDoom multiplayer stuff
                        # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                        #     key = f'PLAYER{player}_FRAGCOUNT'
                        #     if key in infos[0]:
                        #         log.debug('Score for player %d: %r', player, infos[0][key])
                        break

                if all(done) or max_frames_reached(num_frames):
                    break

            if not cfg.no_render:
                env.render()
            time.sleep(0.01)

            episode_rewards.append(episode_reward)
            last_episodes = episode_rewards[-100:]
            avg_reward = sum(last_episodes) / len(last_episodes)
            log.info(
                'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
            )

            if max_frames_reached(num_frames):
                break

    env.close()

    audios = np.array(audios)
    videos = np.array(screens)

    # ran = np.random.randint(200)
    os.makedirs("demo/videos/"+cfg.env, exist_ok=True)

    plot.specgram(audios[:,0])
    plot.savefig('demo/videos/'+ cfg.env +'/specl.png')
    plot.specgram(audios[:,1])
    plot.savefig('demo/videos/'+ cfg.env +'/specr.png')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('demo/videos/'+ cfg.env +'/video.mp4', fourcc, 35/env.skip_frames, (128,72))
    out = cv2.VideoWriter('demo/videos/'+ cfg.env +'/video.mp4', fourcc, 35/(env.skip_frames), (env.screen_w,env.screen_h))
    for i in range(len(screens)):
        out.write(screens[i][:,:,::-1])
    out.release()
    write('demo/videos/'+ cfg.env +'/audio.wav', env.sampling_rate_int, audios)
    # print("total audio time should be :" + str(d))
    my_clip = mpe.VideoFileClip('demo/videos/'+ cfg.env +'/video.mp4')
    audio_background = mpe.AudioFileClip('demo/videos/'+ cfg.env +'/audio.wav')
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("demo/videos/"+ cfg.env +"/movie.mp4")
    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    register_custom_components()
    parser = arg_parser(evaluation = True)
    cfg = parse_args(parser=parser)

    status, avg_reward = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())