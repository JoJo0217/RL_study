import argparse
import torch
import gym
import random
import copy

import numpy as np
from trainer import DQN
from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="dqn")
    parser.add_argument("--output", type=str, default="./model.pth")
    parser.add_argument("--total_episode", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--logging_step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device: ", device)

    logger = Logger(args.logdir)
    logger.log(0, is_tensor_board=False, **vars(args))
    env = gym.make('CartPole-v1')
    agent = DQN(50000, device).to(device)
    score = 0
    target_update = 10
    target = copy.deepcopy(agent)
    score_list = []
    for i_episode in range(args.total_episode):
        if i_episode % target_update == 0:
            target = copy.deepcopy(agent)
        observation = env.reset()

        done = False
        while not done:
            epsilon = max(0.05, 0.25 - 0.01 * i_episode / 100)
            action = agent.action(torch.tensor(observation), epsilon)
            next_observation, reward, done, info = env.step(action)
            done = 1 if done else 0
            agent.input_data((observation, action, reward, next_observation, done))
            observation = next_observation

            score += reward
            if len(agent.buffer) > 2000:
                agent.train(32, target)

            if done == 1:
                break
        if i_episode % args.logging_step == 0:
            logger.log(i_episode, score=score / args.logging_step, epsilon=epsilon)
            score = 0

    env.close()
    torch.save(agent, 'model.pth')


if __name__ == "__main__":
    main()
