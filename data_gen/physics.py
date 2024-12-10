import sys
import os
from pathlib import Path
import numpy as np
import argparse
import h5py

# Add current working directory to path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from envs import physics_sim
from utils import save_list_dict_h5py


def str_to_list(s):
    """Convert string representation of list to actual list of floats"""
    try:
        # Remove brackets and split by comma
        values = s.strip('[]').split(',')
        return [float(x.strip()) for x in values]
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate n-body physics simulation dataset')
    parser.add_argument('--fname', type=str, default='data',
                        help='File name / path.')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes to generate.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Create evaluation set.')
    parser.add_argument('--num-bodies', type=int, default=3, choices=[2, 3],
                        help='Number of bodies to simulate (2 or 3).')
    parser.add_argument('--masses', type=str, default=None,
                        help='Comma-separated masses of the bodies (e.g., "1.0,1.0,1.0").')
    parser.add_argument('--seq-len', type=int, default=12,
                        help='Length of each sequence.')
    parser.add_argument('--img-size', type=int, default=50,
                        help='Size of the square image (width=height).')
    
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Parse masses
    if args.masses is None:
        masses = [1.0] * args.num_bodies
    else:
        masses = str_to_list(args.masses)
        if masses is None or len(masses) != args.num_bodies:
            raise ValueError(f"Invalid masses format. Please provide {args.num_bodies} comma-separated values.")

    # Generate dataset
    physics_sim.generate_nbody_problem_dataset(
        dest=args.fname + '.npz',
        n_bodies=args.num_bodies,
        train_set_size=args.num_episodes,
        valid_set_size=2,
        test_set_size=2,
        seq_len=args.seq_len,
        img_size=[args.img_size, args.img_size],
        masses=masses,
        vx0_max=0.5,
        vy0_max=0.5,
        color=True,
        seed=args.seed
    )

    # Load and process data
    try:
        with np.load(args.fname + '.npz') as data:
            # Shape: (num_samples, num_steps, height, width, channels)
            train_x = data['train_x']
            
            # Create pairs of consecutive frames
            # New shape: (num_samples, num_steps-1, channels*2, height, width)
            train_pairs = np.concatenate(
                (train_x[:, :-1], train_x[:, 1:]), 
                axis=-1
            )
            train_pairs = np.transpose(train_pairs, (0, 1, 4, 2, 3)) / 255.

            # Create replay buffer
            replay_buffer = []
            for idx in range(train_x.shape[0]):
                sample = {
                    'obs': train_pairs[idx, :-1],
                    'next_obs': train_pairs[idx, 1:],
                    'action': np.zeros(train_pairs.shape[1] - 1, dtype=np.int64)
                }
                replay_buffer.append(sample)

            # Save to H5 file
            save_list_dict_h5py(replay_buffer, args.fname)
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()