"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import argparse
import ale_py

# noinspection PyUnresolvedReferences
import envs

import utils

import gymnasium as gym
from gymnasium import logger

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from typing import Dict, List, Tuple, Optional, Any, Union

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space: gym.spaces.Space) -> None:
        self.action_space = action_space

    def act(self, observation: np.ndarray, reward: float, done: bool) -> np.ndarray:
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img: np.ndarray, crop_ratio: Tuple[int, int]) -> np.ndarray:
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Resampling.LANCZOS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    args = parser.parse_args()

    # gym.register_envs(ale_py)

    print("env_id: ", args.env_id)

    env = gym.make(args.env_id)

    # Set random seeds
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes
    reward = 0
    done = False

    crop = None
    warmstart = None
    if args.env_id == 'Pong-ramDeterministic-v4':
        crop = (35, 190)
        warmstart = 58
    elif args.env_id == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warmstart = 50

    if args.atari:
        env._max_episode_steps = warmstart + 11

    replay_buffer = []

    for i in range(episode_count):

        if i % 50 == 0 and i > 0:
            utils.save_list_dict_h5py(replay_buffer, args.fname, offset=i-50)
            replay_buffer = []  # Clear buffer to save memory

        replay_buffer.append({
            'obs': [],
            'action': [],
            'next_obs': [],
        })
        ob, _ = env.reset()

        if args.atari:
            # Burn-in steps
            for _ in range(warmstart):
                action = agent.act(ob, reward, done)
                ob, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            prev_ob = crop_normalize(ob, crop)
            ob, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            ob = crop_normalize(ob, crop)

            while True:
                new_i = i % 50
                replay_buffer[-1]['obs'].append(
                    np.concatenate((ob, prev_ob), axis=0))
                prev_ob = ob

                action = agent.act(ob, reward, done)
                ob, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ob = crop_normalize(ob, crop)

                replay_buffer[-1]['action'].append(action)
                replay_buffer[-1]['next_obs'].append(
                    np.concatenate((ob, prev_ob), axis=0))

                if done:
                    break
        else:
            while True:
                new_i = i % 50
                replay_buffer[-1]['obs'].append(ob[1])

                action = agent.act(ob, reward, done)
                ob, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer[-1]['action'].append(action)
                replay_buffer[-1]['next_obs'].append(ob[1])

                if done:
                    break

        if i % 10 == 0:
            print("iter "+str(i))

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname, offset=episode_count-len(replay_buffer))

if __name__ == '__main__':
    main()