import gym

env = gym.make('MountainCar-v0', render_mode="human")
observation = env.reset()
t = 0
while True:
     t += 1
     env.render()
     observation = env.reset()
     print(observation)
     random_action = env.action_space.sample()
     observation, reward, done, info, _ = env.step(random_action)
     if done:
         print("Episode finished after {} timesteps".format(t+1))
         break

env.close()
