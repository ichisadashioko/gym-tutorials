import time
import random

import numpy as np

import gym


def random_action():
    action = random.choice([0, 1])
    return action


def invert_pole_vel(observation):
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    if pole_vel < 0:
        action = 0  # left
    else:
        action = 1  # right
    return action


def invert_pole_angle(observation):
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    if pole_angle < 0:
        action = 0  # left
    else:
        action = 1  # right
    return action

    return action


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    obs = env.reset()
    for ts in range(1024):
        env.render()

        # cart_pos [-2.4..2.4]
        # cart_vel [-inf..inf]
        # pole_angle [-41.8..41.8]
        # pole_vel [-inf..inf]

        # action = 0 # left only
        # action = 1 # right only
        # action = random_action()
        action = invert_pole_angle(obs)
        # action = invert_pole_vel(obs)

        action_ret = env.step(action)
        obs, reward, done, info = action_ret
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        print(
            f'observation: {obs}',
            f'reward: {reward}',
            f'done: {done}',
            f'info: {info}',
        )

        # if done:
        #     break

        if cart_pos > 2.4 or cart_pos < -2.4:
            break

    env.close()
