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


def normalize_obs(obs):
    return np.transpose(np.array(obs), (2, 0, 1)) / 255.0 - 0.5


def seed_episode(env, replay_buffer, num_episode):
    print("Collecting seed data...")
    for _ in tqdm(range(num_episode)):
        obs, _ = env.reset()
        obs = normalize_obs(obs)
        done = False
        experience = []
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = normalize_obs(next_obs)
            experience.append((obs, np.array(action), reward, next_obs, done))
            obs = next_obs
        for exp in experience:
            replay_buffer.push(*exp)


def collect_data(args, env, obs_shape, action_dim, num_episode, world_model, actor, replay_buffer, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    print("Collecting data...")
    total_reward = 0
    with torch.no_grad():
        for i in tqdm(range(num_episode)):
            obs, info = env.reset()
            obs = normalize_obs(obs)
            done = False
            prev_deter = recurrent.init_hidden(1).to(device)
            prev_state = recurrent.init_state(1).to(device)
            prev_action = torch.zeros(1, action_dim).to(device)
            while not done:
                obs_embed = encoder(torch.tensor(
                    obs, dtype=torch.float32).to(device).unsqueeze(0))
                deter = recurrent(prev_state, prev_action, prev_deter)
                _, posterior = representation(deter, obs_embed)
                _, action = actor(posterior, deter)
                next_obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                next_obs = normalize_obs(next_obs)
                done = terminated or truncated
                replay_buffer.push(obs, action[0].cpu(), np.array(reward), next_obs, done)
                obs = next_obs
                prev_deter = deter
                prev_state = posterior
                prev_action = action
                total_reward += reward
    return total_reward / num_episode


def evaluate(args, env_, obs_shape, action_dim, num_episode, world_model, actor, replay_buffer, device, is_render=True):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    print("Evaluating...")
    total_reward = 0

    if is_render:
        env = gym.wrappers.RecordVideo(env_, video_folder="./video",
                                       episode_trigger=lambda x: x % 1 == 0)
    else:
        env = env_

    with torch.no_grad():
        for i in tqdm(range(num_episode)):
            obs, info = env.reset()
            obs = normalize_obs(obs)
            done = False
            prev_deter = recurrent.init_hidden(1).to(device)
            prev_state = recurrent.init_state(1).to(device)
            prev_action = torch.zeros(1, action_dim).to(device)
            while not done:
                obs_embed = encoder(torch.tensor(
                    obs, dtype=torch.float32).to(device).unsqueeze(0))
                deter = recurrent(prev_state, prev_action, prev_deter)
                _, posterior = representation(deter, obs_embed)
                _, action = actor(posterior, deter)
                next_obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                next_obs = normalize_obs(next_obs)
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


def kl_balance_loss(prior_mean, prior_std, posterior_mean, posterior_std, alpha, freebits):
    prior_dist = torch.distributions.Normal(prior_mean, prior_std)
    prior_dist = torch.distributions.Independent(prior_dist, 1)
    posterior_dist = torch.distributions.Normal(posterior_mean, posterior_std)
    posterior_dist = torch.distributions.Independent(posterior_dist, 1)

    prior_dist_sg = torch.distributions.Normal(prior_mean.detach(), prior_std.detach())
    prior_dist_sg = torch.distributions.Independent(prior_dist_sg, 1)
    posterior_dist_sg = torch.distributions.Normal(posterior_mean.detach(), posterior_std.detach())
    posterior_dist_sg = torch.distributions.Independent(posterior_dist_sg, 1)

    kl_loss = alpha * torch.max(torch.distributions.kl.kl_divergence(posterior_dist_sg, prior_dist).mean(), torch.tensor(freebits)) + \
        (1 - alpha) * torch.max(torch.distributions.kl.kl_divergence(posterior_dist,
                                                                     prior_dist_sg).mean(), torch.tensor(freebits))

    return kl_loss


def train_world(args, batch, world_model, world_optimizer, world_model_params, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    obs_seq, action_seq, reward_seq, next_obs_seq, done_seq = batch

    # (batch, seq, (item))
    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
    action_seq = torch.tensor(action_seq, dtype=torch.float32).to(device)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)
    next_obs_seq = torch.tensor(next_obs_seq, dtype=torch.float32).to(device)
    done_seq = torch.tensor(done_seq, dtype=torch.float32).to(device)
    batch_size = args.batch_size
    seq_len = args.batch_seq

    deter = recurrent.init_hidden(batch_size).to(device)
    state = recurrent.init_state(batch_size).to(device)

    states = torch.zeros(batch_size, seq_len, args.state_size).to(device)
    deters = torch.zeros(batch_size, seq_len, args.deterministic_size).to(device)

    obs_embeded = encoder(obs_seq.view(-1, *obs_seq.size()[2:])
                          ).view(batch_size, seq_len, args.observation_size)
    discount_criterion = nn.BCELoss()

    prior_mean = []
    prior_std = []
    posterior_mean = []
    posterior_std = []

    for t in range(1, seq_len):
        deter = recurrent(state, action_seq[:, t - 1], deter)
        prior_dist, _ = transition(deter)
        posterior_dist, state = representation(deter, obs_embeded[:, t])

        prior_mean.append(prior_dist.mean)
        prior_std.append(prior_dist.scale)
        posterior_mean.append(posterior_dist.mean)
        posterior_std.append(posterior_dist.scale)

        deters[:, t] = deter
        states[:, t] = state

    prior_mean = torch.stack(prior_mean, dim=1)
    prior_std = torch.stack(prior_std, dim=1)
    posterior_mean = torch.stack(posterior_mean, dim=1)
    posterior_std = torch.stack(posterior_std, dim=1)

    kl_loss = kl_balance_loss(prior_mean, prior_std, posterior_mean,
                              posterior_std, args.kl_alpha, args.free_bit)

    obs_pred_dist = decoder(states[:, 1:], deters[:, 1:])
    reward_pred_dist = reward(states[:, 1:], deters[:, 1:])
    discount_pred_dist = discount(states[:, 1:], deters[:, 1:])

    obs_loss = obs_pred_dist.log_prob(obs_seq[:, 1:]).mean()
    reward_loss = reward_pred_dist.log_prob(reward_seq[:, 1:]).mean()
    discount_loss = discount_criterion(discount_pred_dist.probs, 1 - done_seq[:, 1:]).mean()

    total_loss = -obs_loss - reward_loss + discount_loss + args.kl_beta * kl_loss
    world_optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(world_model_params, args.clip_grad)
    world_optimizer.step()

    loss = {"kl_loss": kl_loss.item(), "obs_loss": -obs_loss.item(
    ), "reward_loss": -reward_loss.item(), "discount_loss": discount_loss.item()}
    states = states[:, 1:].detach()
    deters = deters[:, 1:].detach()

    return loss, states, deters


def lambda_return(rewards, values, discounts, gamma, lambda_):

    # rewards: (T, B, 1), values: (T, B, 1), discounts: (T, B, 1)

    T, B, item = rewards.size()
    lambda_return = torch.zeros(*rewards.shape).to(rewards.device)

    lambda_return[-1] = values[-1]
    for t in reversed(range(T - 1)):
        lambda_return[t] = rewards[t + 1] + gamma * discounts[t + 1] * \
            ((1 - lambda_) * values[t + 1] + lambda_ * lambda_return[t + 1])

    return lambda_return


def train_actor_critic(args, states, deters, world_model, actor, critic, target_net, actor_optim, critic_optim, device):
    encoder, recurrent, representation, transition, decoder, reward, discount = world_model
    states = states.reshape(-1, states.size(-1))
    deters = deters.reshape(-1, deters.size(-1))

    imagine_states = []
    imagine_deters = []
    imagine_action_log_probs = []
    imagine_entropy = []

    # horizon만큼 진행
    for t in range(args.horizon):
        epsilon = 1e-6
        action_dist, action = actor(states, deters)
        action = action.clamp(-1 + epsilon, 1 - epsilon)
        action_log_prob = action_dist.log_prob(action)
        entropy = action_dist.base_dist.base_dist.entropy()
        deters = recurrent(states, action, deters)
        _, states = transition(deters)

        imagine_states.append(states)
        imagine_deters.append(deters)
        imagine_action_log_probs.append(action_log_prob)
        imagine_entropy.append(entropy)

    # (horizon, B, (item))
    imagine_states = torch.stack(imagine_states, dim=0)
    imagine_deters = torch.stack(imagine_deters, dim=0)
    imagine_action_log_probs = torch.stack(imagine_action_log_probs, dim=0)
    imagine_entropy = torch.stack(imagine_entropy, dim=0)

    predicted_rewards = reward(imagine_states, imagine_deters).mean
    target_values = target_net(imagine_states, imagine_deters).mean
    discount_pred = discount(imagine_states, imagine_deters).mean

    lambda_return_ = lambda_return(
        predicted_rewards, target_values, discount_pred, args.gamma, args.lambda_)

    actor_loss = -args.reinforce_coef * (imagine_action_log_probs[1:].unsqueeze(-1) * (lambda_return_[:-1] - target_values[:-1]).detach()).mean() -\
        (1 - args.reinforce_coef) * lambda_return_[:-1].mean() -\
        args.entropy_coef * imagine_entropy[1:].mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.clip_grad)
    actor_optim.step()

    value_dist = critic(imagine_states[:-1].detach(), imagine_deters[:-1].detach())
    critic_loss = -torch.mean(value_dist.log_prob(lambda_return_[:-1].detach()))

    critic_optim.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), args.clip_grad)
    critic_optim.step()

    loss = {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    return loss
