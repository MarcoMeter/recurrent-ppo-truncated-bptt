import gym
import numpy as np

class CartPole:
    def __init__(self, mask_velocity = False):
        self._env = gym.make("CartPole-v0")
        # Whether to make CartPole partial observable by masking out the velocity.
        if not mask_velocity:
            self._obs_mask = np.ones(4, dtype=np.float32)
        else:
            self._obs_mask =  np.array([1, 0, 1, 0], dtype=np.float32)

    @property
    def vector_observation_space(self):
        return self._env.observation_space.shape

    @property
    def visual_observation_space(self):
        return None

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        vec_obs = self._env.reset()
        return None, vec_obs * self._obs_mask

    def step(self, action):
        vec_obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return None, vec_obs * self._obs_mask, reward / 100.0, done, info

    def close(self):
        self._env.close()