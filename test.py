import gym
import torch
env = gym.make('CartPole-v1')
agent = torch.load('model.pth')
observation = env.reset()
epsilon = 0
done = False
while not done:
    env.render()
    action = agent.action(torch.tensor(observation), epsilon)
    temp = env.step(action)
    observation, reward, done, info = temp[0], temp[1], temp[2], temp[3]
env.close()
