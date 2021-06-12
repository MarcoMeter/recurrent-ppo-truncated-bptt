from gym.spaces import space
import numpy as np
from gym import spaces

class PocMemoryEnv():
    """
    description # TODO
    """
    def __init__(self, step_size = 0.2, glob = False, freeze_agent = False):
        """
        Arguments:
          env {SimpleMemoryTask} - environment
          step_size {float} -- step size of the agent
          glob {boolean} -- global random positions
          freeze_agent {boolean} -- freezes agent until goal positions become invisible
        """
        self._action_space = spaces.Discrete(2)
        self.freeze_agent = freeze_agent
        self._time_penalty = 0.1
        self._step_size = step_size
        self._min_steps = int(1.0 / self._step_size) + 1
        self._num_show_steps = 2    # this should determine for how long the goal is visible
        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3,),
                dtype = np.float32)

        # Create an array with possible positions
        # Valid local positions are two ticks away from 0.0 and between -0.4 and 0.4
        # Valid global positions are between -1 + step_size and 1 - step_size
        # Clipping has to be applied because step_size is a variable now
        num_steps = int( 0.4 / self._step_size)
        lower = min(- 2.0 * self._step_size, -num_steps * self._step_size) if not glob else -1  + self._step_size
        upper = max( 3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size) if not glob else 1

        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = list(map(lambda x: round(x, 2), self.possible_positions)) # fix floating point errors


    def reset(self, **kwargs):
        """ Let the agent start from a random start position. """
        # sample a random position as starting position
        self._position = np.random.choice(self.possible_positions)
        self._rewards = []
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        self._goals = goals[np.random.permutation(2)]
        obs = np.asarray([self._goals[0], self._position, self._goals[1]])
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        reward = 0.0
        done = False
        info = None

        if self._num_show_steps > self._step_count:
            # Execute action if agent is allowed to move
            self._position += self._step_size * (1 - self.freeze_agent) if action == 1 else -self._step_size * (1 - self.freeze_agent)
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]])

            if self.freeze_agent: # Check if agent is allowed to move
                self._step_count += 1
                self._rewards.append(reward)
                return obs, reward, done, info

        else:
            self._position += self._step_size if action == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0]) # mask out goal information

        # Determine reward and episode termination
        reward = 0.0
        done = False
        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                info = {"success" : True}
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
                info = {"success" : False}
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                info = {"success" : True}
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
                info = {"success" : False}
            done = True
        else:
            reward -= self._time_penalty
        self._rewards.append(reward)

        # Wrap up episode information
        if done:
            info = {**info,
                    "reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        # Increase step count
        self._step_count += 1

        return obs, reward, done, info

    def close(self):
        pass