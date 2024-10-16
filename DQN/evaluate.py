import gym
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model.pth")
    parser.add_argument("--record", type=str, default=None)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    env = gym.make('CartPole-v1')
    if args.record is not None:
        env = gym.wrappers.RecordVideo(env, args.record)
    agent = torch.load(args.model)
    observation = env.reset()
    total_reward = 0
    epsilon = 0
    done = False
    while not done:
        env.render()
        action = agent.action(torch.tensor(observation), epsilon)
        temp = env.step(action)
        observation, reward, done, info = temp[0], temp[1], temp[2], temp[3]
        done = 1 if done else 0
        total_reward += reward
    print("total reward: ", total_reward)
    env.close()


if __name__ == "__main__":
    main()
