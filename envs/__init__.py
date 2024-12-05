"""Environment registration."""

from typing import Dict, Any
import gymnasium as gym
from gymnasium.envs.registration import register

from envs.block_pushing import BlockPushing

register(
    id='ShapesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    id='ShapesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    id='CubesTrain-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    id='CubesEval-v0',
    entry_point='envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

def make_env(name: str, **kwargs: Dict[str, Any]) -> gym.Env:
    """Create environment with specific configuration."""
    return gym.make(name, **kwargs)
