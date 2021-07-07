import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples,ExpertBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

class Expertbuffer(BaseBuffer):
    """
    Expert buffer used in off-policy algorithms like TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
    :param expert_buffer: expert_buffer use as demonstration
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        expert_size=10000,
        expert_observation: np.ndarray=None,
        expert_next_obs: np.ndarray=None,
        expert_rewards: np.ndarray=None,
        expert_actions: np.ndarray=None,
    ):
        super(Expertbuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)


        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.expert_size=expert_size
        #self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.expert_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.expert_size, self.n_envs)  + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.expert_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.expert_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.expert_size, self.n_envs), dtype=np.float32)
        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        self.actions =np.array(expert_actions[0:expert_size]).copy()
        self.rewards = np.array(expert_rewards[0:expert_size]).copy()
        self.dones = np.zeros((self.expert_size, self.n_envs), dtype=np.float32)
        self.next_observations=np.array(expert_next_obs[0:expert_size]).copy()
        self.observations=np.array(expert_observation[0:expert_size]).copy()
        self.actions=self.actions.astype(np.float32)
        self.rewards=self.rewards.astype(np.float32)
        self.next_observations=self.next_observations.astype(np.float32)
        self.observations=self.observations.astype(np.float32)
            #   self.add_expert(expert_obs,expert_act,expert_nex_obs,expert_rew,expert_size)
    #add the expert buffer into the replaybuffer size is the size use for buffer


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) :

        upper_bounds=self.expert_size
        batch_inds=np.random.randint(0,upper_bounds,size=batch_size)

        return self._get_samples(batch_inds,env=env)


    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):



        next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return ExpertBufferSamples(*tuple(map(self.to_torch, data)))


