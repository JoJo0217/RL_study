import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence

from tqdm import tqdm
from models import *
from logger import Logger


env = gym.make('CarRacing-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = env.action_space.shape[0]
obs_shape = env.observation_space.shape
print("action space: ", action_dim, ", obs shape: ", obs_shape, sep='')


def collect_data(env, state_dim, transition_representation, agent, replay_buffer, num_episode, device, training=True):
    print("collecting data...")
    score = 0
    for _ in tqdm(range(num_episode)):
        obs, info = env.reset()
        done = False
        experience = []
        prev_state = torch.zeros(1, state_dim).to(device)
        prev_deter = transition_representation.init_hidden(1).to(device)
        prev_action = torch.zeros(1, action_dim).to(device)
        with torch.no_grad():
            while not done:
                # obs(96x96x3) -> (3x96x96) -> (1x3x96x96)
                obs = torch.tensor(obs, dtype=torch.float32).permute(
                    2, 0, 1).unsqueeze(0).to(device) / 255
                # s_t-1, a_t-1, o_t-1 -> s_t
                posterior_mean, posterior_std, prev_deter = transition_representation.posterior(
                    prev_state, prev_action, prev_deter, obs)
                cur_state = posterior_mean + posterior_std * \
                    torch.normal(0, 1, posterior_mean.size()).to(device)

                action_mu, action_std = agent(cur_state, prev_deter)
                eps = torch.normal(0, 1, (1, action_dim)).to(device)
                if training:
                    cur_action = torch.tanh(action_mu + action_std * eps)
                else:
                    cur_action = torch.tanh(action_mu)
                next_obs, reward, terminated, truncated, info = env.step(
                    cur_action[0].cpu().numpy())
                done = terminated or truncated

                experience.append((np.array(obs.squeeze(0).cpu()), np.array(
                    cur_action.squeeze(0).detach().cpu()), reward, done))

                obs = next_obs
                prev_state = cur_state
                prev_action = cur_action
                score += reward
        if training:
            for exp in experience:
                replay_buffer.push(exp)
    return score / num_episode


def lambda_return(rewards, values, gamma, lambda_):
    # rewards, values : (Horizon+1, seq*batch)
    # 어렵다
    V_lambda = torch.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H + 1):
        # n-step 계산 하기 위함
        # 각 step의 value 목표
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n + 1):
            # n step의 reward 합 진행
            if k == n:
                V_n[:-n] += (gamma ** (n - 1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k - 1)) * rewards[k:-n + k]

        # add lambda_ weighted n-step target to compute lambda target
        if n == H:
            V_lambda += (lambda_ ** (H - 1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n - 1)) * V_n

    return V_lambda


def train(batch, state_dim, deterministic_dim, device, transition_representation, reward_model, observation, actor, value, model_optimizer, actor_optimizer, critic_optimizer):
    obs_seq = []
    action_seq = []
    reward_seq = []
    # batch = batch, seq, (obs, action, reward, done)
    for seq in batch:
        obs_temp = []
        action_temp = []
        reward_temp = []
        for (obs, action, reward, done) in seq:
            obs_temp.append(obs)
            action_temp.append(action)
            reward_temp.append(reward)
        obs_seq.append(obs_temp)
        action_seq.append(action_temp)
        reward_seq.append(reward_temp)
    obs_seq = torch.tensor(obs_seq, dtype=torch.float32).to(device)
    action_seq = torch.tensor(action_seq, dtype=torch.float32).to(device)
    reward_seq = torch.tensor(reward_seq, dtype=torch.float32).to(device)
    batch_size, seq_len, _, _, _ = obs_seq.size()

    prev_deter = transition_representation.init_hidden(batch_size).to(device)
    prev_state = torch.zeros(batch_size, state_dim).to(device)

    states = torch.zeros(seq_len, batch_size, state_dim).to(device)
    deters = torch.zeros(seq_len, batch_size, deterministic_dim).to(device)

    beta = 0.1  # kl조절
    imagine_horizon = 15
    gamma = 0.99
    lambda_ = 0.95
    kl_loss = 0
    reconstruction_loss = 0
    reward_loss = 0

    total_kl_loss = 0
    total_reconstruction_loss = 0
    total_reward_loss = 0

    action_prev = action_seq[:, 0].to(device)
    total_loss = torch.zeros(1).to(device)
    for t in range(1, seq_len):
        obs = obs_seq[:, t].to(device)
        action = action_seq[:, t].to(device)
        reward = reward_seq[:, t].to(device)
        prior_mean, prior_std, _ = transition_representation(prev_state, action_prev, prev_deter)
        posterior_mean, posterior_std, cur_deter = transition_representation.posterior(
            prev_state, action_prev, prev_deter, obs)

        state = posterior_mean + posterior_std * \
            torch.normal(0, 1, posterior_mean.size()).to(device)
        obs_pred = observation(state, cur_deter)
        reconstruction_loss = nn.functional.mse_loss(obs_pred, obs)

        reward_pred = reward_model(state, cur_deter)
        reward_loss = nn.functional.mse_loss(reward_pred, reward)

        prior = Normal(prior_mean, prior_std)
        posterior = Normal(posterior_mean, posterior_std)
        kl_loss = kl_divergence(posterior, prior).mean()

        total_loss += reconstruction_loss + reward_loss + beta * kl_loss

        action_prev = action
        prev_state = state
        prev_deter = cur_deter

        states[t] = state
        deters[t] = cur_deter

        total_kl_loss += kl_loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_reward_loss += reward_loss.item()
    model_optimizer.zero_grad()
    total_loss.backward()
    model_optimizer.step()

    # actor, critic 학습

    # states (seq, batch, state_dim) -> (seq*batch, state_dim)
    # deters (seq, batch, deterministic_dim) -> (seq*batch, deterministic_dim)
    states = states.view(-1, state_dim).detach()
    deters = deters.view(-1, deterministic_dim).detach()

    imagined_states = [states]
    imagined_deters = [deters]

    rewards = []
    values = []

    rewards.append(reward_model(states, deters).squeeze())
    values.append(value(states, deters).squeeze())

    for t in range(1, imagine_horizon + 1):
        action_mu, action_std = actor(imagined_states[t - 1], imagined_deters[t - 1])
        eps = torch.normal(0, 1, (action_mu.size())).to(device)
        action = torch.tanh(action_mu + action_std * eps)

        prior_mean, prior_std, deter = transition_representation(
            imagined_states[t - 1], action, imagined_deters[t - 1])
        state = prior_mean + prior_std * torch.normal(0, 1, prior_mean.size()).to(device)

        imagined_states.append(state)
        imagined_deters.append(deter)

        rewards.append(reward_model(imagined_states[t], imagined_deters[t]).squeeze())
        values.append(value(imagined_states[t], imagined_deters[t]).squeeze())

    imagined_states = torch.stack(imagined_states, dim=0)
    imagined_deters = torch.stack(imagined_deters, dim=0)
    values = torch.stack(values, dim=0)
    rewards = torch.stack(rewards, dim=0)

    returns = lambda_return(rewards, values, 0.99, 0.95)

    critic_loss = nn.functional.mse_loss(values[1:], returns[1:].detach())
    critic_optimizer.zero_grad()
    critic_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=100)
    critic_optimizer.step()

    actor_loss = -returns.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=100)
    actor_optimizer.step()

    print("actor loss: ", actor_loss.item(), ", critic loss: ", critic_loss.item(), sep='')

    return total_kl_loss / (seq_len - 1), total_reconstruction_loss / (seq_len - 1), total_reward_loss / (seq_len - 1), actor_loss.item(), critic_loss.item()


state_dim = 64
deterministic_dim = 256
model_lr = 1e-4
actor_critc_lr = 1e-4
transition_representation = TransitionRepresentationModel(state_dim, action_dim).to(device)
observation = ObservationModel(state_dim, deterministic_dim, obs_shape[2]).to(device)
reward = RewardModel(state_dim, deterministic_dim).to(device)

agent = Agent(state_dim, deterministic_dim, action_dim).to(device)
value = ValueModel(state_dim, deterministic_dim).to(device)

model_params = list(transition_representation.parameters()) + \
    list(observation.parameters()) + list(reward.parameters())
model_optimizer = optim.Adam(model_params, lr=model_lr)
actor_optimizer = optim.Adam(agent.parameters(), lr=actor_critc_lr)
critic_optimizer = optim.Adam(value.parameters(), lr=actor_critc_lr)

# state, action, reward, next_state, done 저장하고 sampling 가능
replay_buffer = ReplayBufferSeq(100000)
logger = Logger('./logs')

num_epochs = 10000
batch_size = 64
seq_len = 50

world_episodes = 1
update_step = 20

seed_episodes = 5
test_interval = 3
save_interval = 20
print("collecting seed data...")
collect_data(env, state_dim, transition_representation, agent, replay_buffer, seed_episodes, device)

for epoch in range(num_epochs):
    train_score = collect_data(env, state_dim, transition_representation,
                               agent, replay_buffer, world_episodes, device)
    logger.log(epoch * update_step, train_score=train_score)

    if len(replay_buffer) < batch_size * seq_len:
        continue

    # train world model and actor, critic
    for _ in range(update_step):
        batch = replay_buffer.sample_seq(batch_size, seq_len)
        kl_loss, reconst_loss, reward_loss, actor_loss, critic_loss = train(
            batch, state_dim, deterministic_dim, device, transition_representation, reward, observation, agent, value, model_optimizer, actor_optimizer, critic_optimizer)
        logger.log(epoch * update_step + _, epoch=epoch, kl_loss=kl_loss, reconst_loss=reconst_loss,
                   reward_loss=reward_loss, actor_loss=actor_loss, critic_loss=critic_loss)

    if epoch % test_interval == 0:
        test_score = collect_data(env, state_dim, transition_representation,
                                  agent, replay_buffer, world_episodes, device, training=False)
        logger.log(epoch * update_step, test_score=test_score)
    if epoch % save_interval == 0:
        torch.save(transition_representation.state_dict(), 'transition_representation.pth')
        torch.save(observation.state_dict(), 'observation.pth')
        torch.save(reward.state_dict(), 'reward.pth')
        torch.save(agent.state_dict(), 'agent.pth')
        torch.save(value.state_dict(), 'value.pth')
torch.save(transition_representation.state_dict(), 'transition_representation.pth')
torch.save(observation.state_dict(), 'observation.pth')
torch.save(reward.state_dict(), 'reward.pth')
torch.save(agent.state_dict(), 'agent.pth')
torch.save(value.state_dict(), 'value.pth')
