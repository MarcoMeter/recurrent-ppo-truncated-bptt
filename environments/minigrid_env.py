import numpy as np
import gym
from gym_minigrid.wrappers import ViewSizeWrapper

class MinigridMemoryVector:
    def __init__(self, view_size = 3):
        self._env = gym.make("MiniGrid-MemoryS7-v0")
        self._env = ViewSizeWrapper(self._env, view_size)
        self._vector_observation_space = (view_size**2*5,)

    @property
    def vector_observation_space(self):
        return self._vector_observation_space

    @property
    def visual_observation_space(self):
        return None

    @property
    def action_space(self):
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        vec_obs = self.process_obs(obs["image"])
        return None, vec_obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        vec_obs = self.process_obs(obs["image"])
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return None, vec_obs, reward, done, info

    def close(self):
        self._env.close()

    def process_obs(self, obs):
        one_hot_obs = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i,j,0] == 1:
                    one_hot_obs.append([0, 1, 0, 0, 0]) # walkable tile
                elif obs[i,j,0] == 2:
                    one_hot_obs.append([0, 0, 1, 0, 0]) # blocked tile
                elif obs[i,j,0] == 5:
                    one_hot_obs.append([0, 0, 0, 1, 0]) # key tile
                elif obs[i,j,0] == 6:
                    one_hot_obs.append([0, 0, 0, 0, 1]) # circle tile
                else:
                    one_hot_obs.append([0, 0, 0, 0, 0]) # anything else
        # return flattened one-hot encoded observation
        return np.asarray(one_hot_obs, dtype=np.float32).reshape(-1)

class MinigridMemoryVisual:
    def __init__(self, view_size = 3):
        self._env = gym.make("MiniGrid-MemoryS7-v0")
        self._env = ViewSizeWrapper(self._env, 3)
        self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, 84, 84),
                dtype = np.float32)

    @property
    def vector_observation_space(self):
        return None

    @property
    def visual_observation_space(self):
        return self._visual_observation_space

    @property
    def action_space(self):
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        vis_obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255.
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)
        return vis_obs, None

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        vis_obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255.
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)
        return vis_obs, None, reward, done, info

    def close(self):
        self._env.close()