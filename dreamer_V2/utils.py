import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_episode(env, replay_buffer, num_episode):
    print("collecting seed data...")
    for _ in tqdm(range(num_episode)):
        obs, _ = env.reset()
        done = False
        experience = []
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            experience.append((np.array(obs), np.array(action), reward, done))
            obs = next_obs
        for exp in experience:
            replay_buffer.push(*exp)


def collect_data(args, env, obs_shape, action_dim, num_episode, world_model, actor, replay_buffer, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    print("Collecting data")
    total_reward = 0
    with torch.no_grad():
        for i in tqdm(range(num_episode)):
            obs, info = env.reset()
            done = False
            prev_deter = recurrent.init_hidden(1).to(device)
            prev_state = recurrent.init_state(1).to(device)
            prev_action = torch.zeros(1, action_dim).to(device)
            while not done:
                obs = np.array(obs).reshape(-1)
                obs_embed = encoder(torch.tensor(
                    obs, dtype=torch.float32).to(device).view(1, obs_shape))
                deter = recurrent(prev_action, prev_state, prev_deter)
                _, posterior = representation(deter, obs_embed)
                action_dist = actor(posterior, deter)
                action = action_dist.sample()
                action = torch.tanh(action)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = terminated or truncated
                reward = np.array(reward)
                done = np.array(done)
                replay_buffer.push(obs, action.cpu(), reward, done)
                obs = next_obs
                prev_deter = deter
                prev_state = posterior
                prev_action = action
                total_reward += reward
    return total_reward / num_episode


def evaluate(args, env_, obs_shape, action_dim, num_episode, world_model, actor, replay_buffer, device, is_render=True):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    print("Collecting data")
    total_reward = 0

    if is_render:
        env = gym.wrappers.RecordVideo(env_, video_folder="./video",
                                       episode_trigger=lambda x: x % 1 == 0)
    else:
        env = env_

    with torch.no_grad():
        for i in tqdm(range(num_episode)):
            obs, info = env.reset()
            done = False
            prev_deter = recurrent.init_hidden(1).to(device)
            prev_state = recurrent.init_state(1).to(device)
            prev_action = torch.zeros(1, action_dim).to(device)
            while not done:
                obs = np.array(obs).reshape(-1)
                obs_embed = encoder(torch.tensor(
                    obs, dtype=torch.float32).to(device).view(1, obs_shape))
                deter = recurrent(prev_action, prev_state, prev_deter)
                _, posterior = representation(deter, obs_embed)
                action = actor(posterior, deter, training=False)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = terminated or truncated
                reward = np.array(reward)

                obs = next_obs
                prev_deter = deter
                prev_state = posterior
                prev_action = action
                total_reward += reward
    return total_reward / num_episode


def save_model(args, world_model, actor, critic):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    os.makedirs(args.output, exist_ok=True)
    torch.save(encoder.state_dict(), args.output + "/encoder.pth")
    torch.save(recurrent.state_dict(), args.output + "/recurrent.pth")
    torch.save(representation.state_dict(), args.output + "/representation.pth")
    torch.save(transition.state_dict(), args.output + "/transition.pth")
    torch.save(decoder.state_dict(), args.output + "/decoder.pth")
    torch.save(reward.state_dict(), args.output + "/reward.pth")
    torch.save(discount.state_dict(), args.output + "/discount.pth")
    torch.save(actor.state_dict(), args.output + "/actor.pth")
    torch.save(critic.state_dict(), args.output + "/critic.pth")


def train_world(args, batch, world_model, world_optimizer, world_model_params, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    obs_seq, action_seq, reward_seq, done_seq = batch

    # (batch, seq, (item))
    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
    action_seq = torch.tensor(action_seq, dtype=torch.float32).to(device)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)
    done_seq = torch.tensor(done_seq, dtype=torch.float32).to(device)
    batch_size = args.batch_size
    seq_len = args.batch_seq

    prev_deter = recurrent.init_hidden(batch_size).to(device)
    prev_state = recurrent.init_state(batch_size).to(device)

    states = torch.zeros(batch_size, seq_len, args.state_size).to(device)
    deters = torch.zeros(batch_size, seq_len, args.deterministic_size).to(device)

    obs_embeded = encoder(obs_seq.view(-1, obs_seq.size(-1))
                          ).view(batch_size, seq_len, args.observation_size)
    discount_criterion = nn.BCELoss()
    kl_loss = torch.zeros(batch_size).to(device)
    for t in range(1, seq_len):
        deter = recurrent(prev_state, action_seq[:, t - 1], prev_deter)
        prior_dist, prior = transition(deter)
        posterior_dist, posterior = representation(deter, obs_embeded[:, t])

        prior_dist_sg, _ = transition.stop_grad(deter)
        posterior_dist_sg, _ = representation.stop_grad(deter, obs_embeded[:, t])

        kl_loss += args.kl_alpha * \
            torch.distributions.kl.kl_divergence(prior_dist, posterior_dist_sg).sum(-1)
        kl_loss += (1 - args.kl_alpha) * \
            torch.distributions.kl.kl_divergence(prior_dist_sg, posterior_dist).sum(-1)
        deters[:, t] = deter
        states[:, t] = prior

        prev_deter = deter
        prev_state = posterior

    kl_loss = kl_loss / (seq_len - 1)

    obs_pred_dist = decoder(states[:, 1:], deters[:, 1:])
    reward_pred_dist = reward(states[:, 1:], deters[:, 1:])
    discount_pred_dist = discount(states[:, 1:], deters[:, 1:])

    obs_loss = obs_pred_dist.log_prob(obs_seq[:, 1:]).mean()
    reward_loss = reward_pred_dist.log_prob(reward_seq[:, 1:]).mean()
    discount_loss = discount_criterion(discount_pred_dist.probs, 1 - done_seq[:, 1:]).mean()

    total_loss = -obs_loss - reward_loss + discount_loss + args.kl_beta * kl_loss.mean()
    world_optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(world_model_params, args.clip_grad)
    world_optimizer.step()

    loss = {"kl_loss": kl_loss.mean().item(), "obs_loss": -obs_loss.item(
    ), "reward_loss": -reward_loss.item(), "discount_loss": discount_loss.item()}
    states = states[:, 1:].detach()
    deters = deters[:, 1:].detach()

    return loss, states, deters


def lambda_return(rewards, values, discounts, lambda_):

    # rewards: (T, B), values: (T, B), discounts: (T, B)

    T, B = rewards.size()
    lambda_return = torch.zeros(T, B).to(rewards.device)

    lambda_return[-1] = rewards[-1] + discounts[-1] * values[-1]
    for t in reversed(range(T - 1)):
        lambda_return[t] = rewards[t] + discounts[t] * \
            ((1 - lambda_) * values[t + 1] + lambda_ * lambda_return[t + 1])

    return lambda_return


def train_actor_critic(args, states, deters, world_model, actor, critic, target_net, actor_optim, critic_optim, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    states = states.reshape(-1, states.size(-1))
    deters = deters.reshape(-1, deters.size(-1))

    imagine_states = [states]
    imagine_deters = [deters]
    imagine_rewards = []
    imagine_values = []
    imagine_values_target = []
    imagine_action_log_probs = []
    imgaine_discounts = []
    imgaine_entropy = []

    for t in range(1, args.horizon + 1):
        action_dist = actor(imagine_states[t - 1], imagine_deters[t - 1])
        action = action_dist.rsample()
        action_log_prob = action_dist.log_prob(action).sum(-1)
        action = torch.tanh(action)
        entropy = action_dist.entropy().sum(-1)
        deter = recurrent(imagine_states[t - 1], action, imagine_deters[t - 1])
        _, prior = transition(deter)

        imagine_states.append(prior)
        imagine_deters.append(deter)

        reward_dist = reward(imagine_states[t], imagine_deters[t])
        reward_pred = reward_dist.sample()
        discount_dist = discount(imagine_states[t], imagine_deters[t])
        discount_pred = discount_dist.sample()

        value = critic(imagine_states[t], imagine_deters[t])
        target_value = target_net(imagine_states[t], imagine_deters[t])

        imagine_action_log_probs.append(action_log_prob)
        imagine_rewards.append(reward_pred)
        imgaine_discounts.append(discount_pred)
        imagine_values.append(value)
        imagine_values_target.append(target_value)
        imgaine_entropy.append(entropy)

    imagine_rewards = torch.stack(imagine_rewards, dim=0).squeeze(-1)
    imgaine_discounts = torch.stack(imgaine_discounts, dim=0).squeeze(-1)
    imagine_values = torch.stack(imagine_values, dim=0).squeeze(-1)
    imagine_values_target = torch.stack(imagine_values_target, dim=0).squeeze(-1)
    imagine_action_log_probs = torch.stack(imagine_action_log_probs, dim=0).squeeze(-1)

    lambda_return_ = lambda_return(
        imagine_rewards, imagine_values_target, imgaine_discounts, args.lambda_)
    print(lambda_return_.mean(dim=-1), imagine_values.mean(dim=-1))
    critic_loss = nn.functional.mse_loss(imagine_values[:-1], lambda_return_[:-1].detach())

    actor_loss = -args.reinforce_coef * (imagine_action_log_probs[:-1] * (lambda_return_[:-1] - imagine_values_target[:-1]).detach()).mean() -\
        (1 - args.reinforce_coef) * lambda_return_[:-1].mean() -\
        args.entropy_coef * torch.stack(imgaine_entropy[:-1], dim=0).mean()

    actor_optim.zero_grad()
    critic_optim.zero_grad()
    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.clip_grad)
    nn.utils.clip_grad_norm_(critic.parameters(), args.clip_grad)
    actor_optim.step()
    critic_optim.step()

    loss = {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    return loss
