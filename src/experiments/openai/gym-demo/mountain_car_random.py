import gym
import random
import time

MAX_EPISODE_TIMESTEPS = 200
env = gym.make('MountainCar-v0', render_mode="human")


def Random_games():
    # each of these episodes is another game
    for episode in range(10):
        env.reset()

        # perform up to MAX_EPISODE_TIMESTEPS, generating that number of frames
        t = 0
        while t < MAX_EPISODE_TIMESTEPS:

            # displays the ennvironment in the current frame
            env.render()

            # create a sample random action in any environment
            random_action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info, _ = env.step(random_action)

            t = t + 1

            print(next_state, reward, done, info, random_action)
            if done:
                break

        time.sleep(1)
    env.close()

Random_games()

