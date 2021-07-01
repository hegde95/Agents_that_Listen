import datetime
import math
import os
import sys
import time
from collections import deque
from os.path import join

import numpy as np
import torch

from algorithms.appo.actor_worker import transform_dict_observations
from algorithms.appo.learner import LearnerWorker
from algorithms.appo.model import create_actor_critic
from algorithms.appo.model_utils import get_hidden_size
from algorithms.utils.action_distributions import ContinuousActionDistribution
from algorithms.utils.algo_utils import ExperimentStatus
from algorithms.utils.arguments import parse_args, load_from_checkpoint
from algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from envs.create_env import create_env
from utils.utils import log, AttrDict
import json

import cv2
from scipy.io.wavfile import write
import moviepy.editor as mpe
import matplotlib.pyplot as plot
from PIL import Image

# python -m demo.audio_enjoy_multiagent_appo --env doom_duel_sound --train_dir train_dir_for_demo --experiment 00_duel_with_sound_self_play_ppo_1 --algo APPO --experiments_root ./

AGENTS = ["Sound", "Sound_Deaf", "Baseline"]

AGENT1 = "Sound"
AGENT2 = "Baseline"
MAKE_VIDEO = True
MAX_EP = 1
baseline_dir = '/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo/00_duel_without_sound_self_play_ppo_1/checkpoint_p0'

def enjoy(cfg, max_num_episodes=MAX_EP, max_num_frames=1e9):
    train_dir = cfg.train_dir
    cfg = load_from_checkpoint(cfg)
    cfg.train_dir = train_dir
    # cfg.train_dir = '/home/khegde/Desktop/Github2/sample-factory/train_dir'

    cfg.device = 'cpu'
    cfg.record_to = None

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    cfg.no_render = True

    if cfg.record_to:
        tstamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        cfg.record_to = join(cfg.record_to, f'{cfg.experiment}', tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    # env.seed(0)

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
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    # baseline model
    actor_critic_b = create_actor_critic(cfg, env.observation_space, env.action_space)

    actor_critic_b.model_to_device(device)

    # checkpoints_b = [
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000490805_2010337280.pth',
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000491042_2011308032.pth',
    #     '/home/khegde/Desktop/Github2/sample-factory/train_dir_for_demo/00_duel_without_sound_self_play_ppo_1/checkpoint_p0/checkpoint_000491077_2011451392.pth'
    # ]
    if AGENT2 == "Baseline":
        checkpoints_b = LearnerWorker.get_checkpoints(baseline_dir)
        print(checkpoints_b)
    elif AGENT2 in ["Sound", "Sound_Deaf"]:
        checkpoints_b = checkpoints

    checkpoint_dict_b = LearnerWorker.load_checkpoint(checkpoints_b, device)
    actor_critic_b.load_state_dict(checkpoint_dict_b['model'])    

    episode_rewards = []
    audios = []
    screens = []
    true_rewards = deque([], maxlen=100)
    num_frames = 0

    player1_score = 0
    player2_score = 0
    draws = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()

    with torch.no_grad():
        for _ in range(max_num_episodes):
            done = [False] * len(obs)
            infos = None
            audio = None
            rnn_states = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)
            rnn_states_b = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)

            episode_reward = 0

            while True:
                obs_torch = AttrDict(transform_dict_observations([obs[0]]))
                obs_torch_b = AttrDict(transform_dict_observations([obs[1]]))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()
                for key, x in obs_torch_b.items():
                    obs_torch_b[key] = torch.from_numpy(x).to(device).float()

                # obs_torch['sound'][1] = torch.zeros(obs_torch['sound'][1].shape)
                # obs_torch['obs'][1] = torch.zeros(obs_torch['obs'][1].shape)
                if AGENT1 == "Sound_Deaf":
                    obs_torch['sound'] = torch.zeros(obs_torch['sound'].shape)
                
                if AGENT2 == "Sound_Deaf":
                    obs_torch_b['sound'] = torch.zeros(obs_torch_b['sound'].shape)
                policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)
                policy_outputs_b = actor_critic_b(obs_torch_b, rnn_states_b, with_action_distribution=True)

                # sample actions from the distribution by default
                actions = policy_outputs.actions
                actions_b = policy_outputs_b.actions

                action_distribution = policy_outputs.action_distribution
                if isinstance(action_distribution, ContinuousActionDistribution):
                    if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                        actions = action_distribution.means

                actions = actions.cpu().numpy()
                actions_b = actions_b.cpu().numpy()

                rnn_states = policy_outputs.rnn_states
                rnn_states_b = policy_outputs_b.rnn_states
                # if infos:
                #     audio = infos[0]['sound_buffer_raw']
                #     if audio is not None:
                #         # screen = env.unwrapped.state.screen_buffer
                #         # scrn = np.swapaxes(np.swapaxes(screen,0,1),1,2)
                #         # screens.append(scrn)                
                #         list_audio = list(audio)
                #         audios.extend(list_audio)

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

                    obs, rew, done, infos = env.step(np.array([actions[0], actions_b[0]]))
                    if 'sound_buffer_raw' in infos[0].keys() and MAKE_VIDEO:
                        audio = infos[0]['sound_buffer_raw']
                        if audio is not None:
                            list_audio = list(audio)
                            # audios.extend(list_audio[:len(list_audio)//2])
                            audios.extend(list_audio[:len(list_audio)//4])

                            screen1 = infos[0]['image_buffer_raw']
                            screen2 = infos[1]['image_buffer_raw']
                            scrn = np.swapaxes(np.swapaxes(screen1,0,1),1,2)
                            scrn = cv2.resize(scrn, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
                            screens.append(scrn)

                    episode_reward += np.mean(rew)
                    num_frames += 1

                    if all(done):
                        true_rewards.append(infos[0].get('true_reward', math.nan))
                        log.info('Episode finished at %d frames', num_frames)
                        if not math.isnan(np.mean(true_rewards)):
                            log.info('true rew %.3f avg true rew %.3f', true_rewards[-1], np.mean(true_rewards))

                        if infos[0].get('FRAGCOUNT') > infos[1].get('FRAGCOUNT'):
                            player1_score += 1
                            log.info( AGENT1 +' agent won!!')
                        elif infos[1].get('FRAGCOUNT') > infos[0].get('FRAGCOUNT'):
                            player2_score += 1
                            log.info( AGENT2 +' agent won!!')
                        else:
                            draws += 1
                            log.info('Draw!!')

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
    # log.info(
    #         'Final score: ' + AGENT1 + ' {} - ' + AGENT2 + ' {} over {} episodes'.format(player1_score, player2_score, max_num_episodes)
    #     )
    result_log = {
        AGENT1 : player1_score,
        AGENT2 : player2_score,
        "DRAW" : draws,
        
    }


    print(result_log)

    # ran = np.random.randint(200)
    os.makedirs("demo/videos/"+cfg.env, exist_ok=True)
    log_dir = 'demo/videos/'+ cfg.env + "/" +  AGENT1 + "-vs-" + AGENT2 + "_ep_" + str(max_num_episodes)
    os.makedirs(log_dir, exist_ok=True)
    # " + AGENT1 + "-vs-" + AGENT2 + "
    if MAKE_VIDEO:
        audios = np.array(audios)
        videos = np.array(screens)
        plot.specgram(audios[:,0])
        plot.savefig(log_dir +'/specl.png')
        plot.specgram(audios[:,1])
        plot.savefig(log_dir +'/specr.png')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('demo/videos/'+ cfg.env +'/video.mp4', fourcc, 35/env.skip_frames, (128,72))
        out = cv2.VideoWriter(log_dir+'/video.mp4', fourcc, 35/(env.skip_frames), (1280, 720))
        for i in range(len(screens)):
            out.write(screens[i][:,:,::-1])
        out.release()
        write(log_dir +'/audio.wav', 22050, audios)
        # print("total audio time should be :" + str(d))
        my_clip = mpe.VideoFileClip(log_dir +'/video.mp4')
        audio_background = mpe.AudioFileClip(log_dir +'/audio.wav')
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(log_dir +"/movie.mp4")
    result_json = json.dumps(result_log)
    f = open(log_dir +"/results.json","w")
    f.write(result_json)
    f.close()
    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    status, avg_reward = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())