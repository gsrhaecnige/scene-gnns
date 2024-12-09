"""Utility functions."""

import os
import h5py
import numpy as np
from typing import Dict, List, Union, Any, Optional

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # Handle case where bias might be None
            nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict: Dict[str, np.ndarray], fname: str) -> None:
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname: str) -> Dict[str, np.ndarray]:
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict: List[Dict[str, np.ndarray]], fname: str, offset: int = 0) -> None:
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Use 'a' (append) mode if file exists, 'w' (write) mode if it doesn't
    mode = 'a' if os.path.exists(fname) else 'w'
    
    with h5py.File(fname, mode) as hf:
        for i in range(len(array_dict)):
            grp_name = str(i + offset)
            # Delete group if it already exists (in append mode)
            if grp_name in hf:
                del hf[grp_name]
            grp = hf.create_group(grp_name)
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname: str) -> List[Dict[str, np.ndarray]]:
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            # print("group: " + str(i))
            array_dict.append(dict())
            for key in hf[grp].keys():
                print(f"Dataset {key}: shape={hf[grp][key].shape}, size={hf[grp][key].size}")
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


def get_colors(cmap: str = 'Set1', num_colors: int = 9) -> List[tuple]:
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn: str) -> nn.Module:
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices: torch.Tensor, max_index: int) -> torch.Tensor:
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array: np.ndarray) -> np.ndarray:
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file: str):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode: List[tuple] = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> tuple:
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        return obs, action, next_obs


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer."""

    def __init__(self, hdf5_file: str, path_length: int = 5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length

    def __len__(self) -> int:
        return len(self.experience_buffer)

    def __getitem__(self, idx: int) -> tuple:
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions
