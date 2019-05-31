import time
import random

import numpy as np

import gym

env = gym.make('CartPole-v0')
observation = env.reset()

for ts in range(64):
    env.render()

    action = int(input('left (0) or right (1): '))

    action_ret = env.step(action)
    obs, reward, done, info = action_ret
    print(f'observation: {obs}, reward: {reward}, done: {done}, info: {info}')

    # if done:
    #     break

env.close()
