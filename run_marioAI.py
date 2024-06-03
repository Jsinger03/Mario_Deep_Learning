import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import TimeLimit, StepAPICompatibility
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnasium.wrappers.vector_list_info import VectorListInfo
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack, GrayScaleObservation
from matplotlib import pyplot as plt
from gymnasium.spaces import Box as GymnasiumBox
import gymnasium.spaces
from custom_wrappers import CustomDummyVecEnv, CustomVecFrameStack, CustomResetWrapper, CustomJoypadSpace, SetNumEnvsWrapper
from stable_baselines3 import PPO

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

os.environ['DISPLAY'] = ':0' #change depending on if on local or on remote

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = unwrap_env(env)
env = EnvCompatibility(env, render_mode='human')
env = StepAPICompatibility(env, output_truncation_bool=True)
env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)
env.observation_space = convert_observation_space(env.observation_space)
env = TimeLimit(env, max_episode_steps=10000)
env = GrayScaleObservation(env, keep_dim=True)
env = SetNumEnvsWrapper(env, num_envs=1)
env = CustomDummyVecEnv([lambda: env])  # Use CustomDummyVecEnv here
env = CustomVecFrameStack(env, n_stack=4, channels_order='last')  # Use CustomVecFrameStack here
env = CustomResetWrapper(env)

model = PPO.load("training_folder/ppo_model_#_steps.zip")



obs, info = env.reset() 
done = False
#while not done:
for step in range(1000000):
    action, obs = model.predict(obs)  # Take a random action
    result = env.step(action)
    
    obs, reward, terminated, truncated, info = result
    done = terminated  # Use 'terminated' to control the loop
    if done:
       obs, info =  env.reset()
    env.render()
