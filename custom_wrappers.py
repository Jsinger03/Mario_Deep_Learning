import numpy as np
from copy import deepcopy
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gymnasium as gym


class CustomDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.buf_infos = {i: {} for i in range(self.num_envs)}  # Initialize buf_infos as a dictionary of dictionaries

    def step_async(self, actions):
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        self.actions = actions
        #print(f"CustomDummyVecEnv step_async - actions: {actions}, type: {type(actions)}")
        assert isinstance(actions, (list, np.ndarray)), f"Expected list or np.ndarray, got {type(actions)}"

    def step_wait(self):
        #print("CustomDummyVecEnv step_wait called")
        actions = self.actions
        #print(f"Actions type: {type(actions)}, Actions: {actions}")
        for env_idx in range(self.num_envs):
            action = self.actions[env_idx]
            #print(f"step_wait - action for env {env_idx}: {action}")
            obs, self.buf_rews[env_idx], terminated, truncated, info = self.envs[env_idx].step(action)
            self.buf_dones[env_idx] = terminated or truncated

            # Update buf_infos dictionary
            self.buf_infos[env_idx].update(info)

            # Initialize and update the "TimeLimit.truncated" key
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                obs, reset_info = self.envs[env_idx].reset()
                self.buf_infos[env_idx]["terminal_observation"] = obs
                self.buf_infos[env_idx].update(reset_info)

            self._save_obs(env_idx, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)


import pdb
class CustomVecFrameStack(VecFrameStack):
    def step(self, action):
        #print("CustomVecFrameStack step")
        observation, reward, done, info = self.venv.step(action)
        observation, info = self.stacked_obs.update(observation, done, info)
        return observation, reward, done, False, info  # Adding False for `truncated`

    def step_wait(self):
        #print("CustomVecFrameStack step_wait called")
        observation, reward, done, info = self.venv.step_wait()
        observation, info = self.stacked_obs.update(observation, done, info)
        result = (observation, reward, done, False, info)  # Adding False for `truncated`
        #print(f"CustomVecFrameStack step_wait result: {result}")
        return result
    
    def reset(self, **kwargs):
        #pdb.set_trace()
        #print("VecFrameStack reset with kwargs:", kwargs)
        # Conditionally pass seed if the wrapped env supports it
        if 'seed' in kwargs and hasattr(self.venv, 'reset'):
            try:
                observation = self.venv.reset(seed=kwargs['seed'])
            except TypeError:
                observation = self.venv.reset()
        else:
            observation = self.venv.reset()
        self.stacked_obs.reset(observation)
        return self.stacked_obs.stacked_obs, {}
        #return super().reset(**kwargs)
# Custom wrapper to ensure reset returns a tuple
class CustomResetWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        #print("CustomResetWrapper reset")
        #print("CustomResetWrapper reset with kwargs:", kwargs)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return obs, info
# Custom JoypadSpace to handle rendering and step method compatibility
class CustomJoypadSpace(gym.Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env: gym.Env, actions: list):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(actions))
        self._action_map = {}
        self._action_meanings = {}
        for action, button_list in enumerate(actions):
            byte_action = 0
            for button in button_list:
                byte_action |= self._button_map[button]
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

    def step(self, action):
        result = self.env.step(self._action_map[action])
        if len(result) == 4:
            observation, reward, done, info = result
            terminated = done
            truncated = False
        elif len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            raise ValueError(f"Unexpected number of values returned from step: {len(result)}")
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
            return observation, info
        elif isinstance(result, (list, np.ndarray)):
            observation = result
            info = {}
            return observation, info
        else:
            raise ValueError(f"Unexpected number of values returned from reset: {len(result)}")

    def get_keys_to_action(self):
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        keys_to_action = {}
        for action, byte in self._action_map.items():
            keys = action_to_keys[byte]
            keys_to_action[keys] = action
        return keys_to_action

    def get_action_meanings(self):
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]
def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

def convert_observation_space(observation_space):
    #print(f"Converting observation space of type: {type(observation_space)}")
    if isinstance(observation_space, GymnasiumBox):
        return observation_space
    else:
        return GymnasiumBox(
            low=observation_space.low,
            high=observation_space.high,
            shape=observation_space.shape,
            dtype=observation_space.dtype
        )
    
class SetNumEnvsWrapper(gym.Wrapper):
    def __init__(self, env, num_envs):
        super().__init__(env)
        self.num_envs = num_envs