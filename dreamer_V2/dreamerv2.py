import torch
import torch.optim as optim
import numpy as np
import random
import argparse
import gymnasium as gym

from tqdm import tqdm
from logger import Logger
from models import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--total_episode", type=int, default=4000)
    parser.add_argument("--seed_episode", type=int, default=5)
    parser.add_argument("--collect_episode", type=int, default=1)
    parser.add_argument("--train_step", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--batch_seq", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--logging_step", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=3)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic_size", type=int, default=256)
    parser.add_argument("--state_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--observation_size", type=int, default=128)
    parser.add_argument("--categorical_size", type=int, default=16)
    parser.add_argument("--class_size", type=int, default=16)
    parser.add_argument("--discrete_action", type=bool, default=True)
    parser.add_argument("--continuous_action", type=bool, default=True)
    parser.add_argument("--model_lr", type=float, default=1e-4)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--kl_beta", type=int, default=1)
    parser.add_argument("--kl_alpha", type=float, default=0.8)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--entropy_coef", type=float, default=1e-4)
    parser.add_argument("--reinforce_coef", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=100)
    args = parser.parse_args(args=[])

    return args


def main():
    args = parse_args()
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 1
    obs_shape = env.observation_space.shape[0]
    print("action_dim:", action_dim, "obs_shape:", obs_shape)

    encoder = Encoder2D(args, obs_shape).to(device)
    recurrent = RSSM(args, action_dim).to(device)
    representation = RepresentationModel(args).to(device)
    transition = TransitionModel(args).to(device)
    decoder = Decoder2D(args, obs_shape).to(device)
    reward = RewardModel(args).to(device)
    discount = DiscountModel(args).to(device)

    model_params = list(encoder.parameters()) + list(recurrent.parameters()) + \
        list(representation.parameters()) + list(transition.parameters()) + \
        list(decoder.parameters()) + list(reward.parameters()) + list(discount.parameters())

    actor = ActionContinuous(args, action_dim).to(device)
    critic = Value(args).to(device)
    target_net = Value(args).to(device)
    for param_p, paran_k in zip(target_net.parameters(), critic.parameters()):
        param_p.data.copy_(paran_k.data)
        param_p.requires_grad = False

    model_optim = optim.Adam(model_params, lr=args.model_lr)
    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)

    replay_buffer = ReplayBufferSeq(100000, (obs_shape,), action_dim)
    logger = Logger(args.logdir)

    world_model = (encoder, recurrent, representation, transition, decoder, reward, discount)

    seed_episode(env, replay_buffer, args.seed_episode)
    for episode in range(args.total_episode):
        batch = replay_buffer.sample(args.batch_size, args.batch_seq)
        print(len(replay_buffer))
        for step in range(args.train_step):
            loss, states, deters = train_world(
                args, batch, world_model, model_optim, model_params, device)
            logger.log(episode * args.train_step + step, epoch=episode, **loss)
            loss = train_actor_critic(args, states, deters, world_model,
                                      actor, critic, target_net, actor_optim, critic_optim, device)
            logger.log(episode * args.train_step + step, epoch=episode, **loss)
        train_score = collect_data(args, env, obs_shape, action_dim, args.collect_episode,
                                   world_model, actor, replay_buffer, device)
        logger.log(episode * args.train_step + step, epoch=episode, train_score=train_score)
        if episode % args.eval_step == 0:
            test_score = evaluate(args, env, obs_shape, action_dim, 1,
                                  world_model, actor, replay_buffer, device, is_render=True)
            logger.log(episode * args.train_step + step, epoch=episode, test_score=test_score)

        if episode % args.save_step == 0:
            save_model(args, world_model, actor, critic)


if __name__ == "__main__":
    main()
