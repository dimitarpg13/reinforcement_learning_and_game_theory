import gym
import random

MAX_EPISODE_TIMESTEPS = 10000
env = gym.make("LunarLander-v2",render_mode="human")

def Random_games():
    # each of these episodes is a new game
    for episode in range(10):
        env.reset()

        # perform up to MAX_EPISODE_TIMESTEPS, generating that number of frames
        t = 0
        while t < MAX_EPISODE_TIMESTEPS:
            # This will display the environment
            # Only display if you really want to see it
            # Takes much longer to display it
            env.render()

            # This will just create a sample action in any environment
            # in this environment, the action can be any of one in 4 action list, for example [0 1 0 0]
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info, _ = env.step(action)

            t = t + 1

            # lets print everything in one line:
            print(next_state, reward, done, info, action)
            if done:
                break

    env.close()

Random_games()
