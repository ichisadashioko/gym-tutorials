import gym

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    for i_eps in range(20):
        observation = env.reset()

        for t in range(100):
            env.render()
            # print(type(observation), observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(
                f'observation: {observation}, reward: {reward}, done: {done}, info: {info}')
            if done:
                print(f'Episode finished after {t+1} timesteps')
                break

    env.close()
