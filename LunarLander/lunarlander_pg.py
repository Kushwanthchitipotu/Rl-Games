#!/usr/bin/env python3
# lunarlander_pg.py
# Runnable policy-gradient script for LunarLander (gymnasium)
import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def compute_reward_to_go(rewards, gamma=0.99):
    rtg = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = running * gamma + rewards[t]
        rtg[t] = running
    return rtg

def set_seed(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

def collect_episodes(env, policy, batch_size, gamma):
    batch_episode_rewards = []
    batch_episode_states = []
    batch_episode_actions = []

    for _ in range(batch_size):
        state, _ = env.reset()
        done = False
        ep_rewards, ep_states, ep_actions = [], [], []

        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32)
            probs = policy(s_tensor)
            action = torch.multinomial(probs, 1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_states.append(s_tensor)
            ep_actions.append(action)
            ep_rewards.append(reward)

            state = next_state

        batch_episode_rewards.append(ep_rewards)
        batch_episode_states.append(ep_states)
        batch_episode_actions.append(ep_actions)

    return batch_episode_states, batch_episode_actions, batch_episode_rewards

def policy_gradient_update(policy, optimizer, batch_states, batch_actions, batch_rewards,
                           reward_to_go=True, advantage_norm=True, gamma=0.99):
    flat_states = [s for ep in batch_states for s in ep]
    flat_actions = [a for ep in batch_actions for a in ep]

    returns_per_timestep = []
    episode_sums = []
    for ep_rewards in batch_rewards:
        episode_sums.append(sum(ep_rewards))
        if reward_to_go:
            returns_per_timestep.extend(compute_reward_to_go(ep_rewards, gamma).tolist())
        else:
            G = sum(ep_rewards)
            returns_per_timestep.extend([G] * len(ep_rewards))

    returns = np.array(returns_per_timestep, dtype=np.float32)
    if advantage_norm and returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    optimizer.zero_grad()
    loss = 0.0
    for s, a, R in zip(flat_states, flat_actions, returns_tensor):
        probs = policy(s)
        logp = torch.log(probs[a] + 1e-8)
        loss -= logp * R

    loss = loss / max(1, len(flat_states))
    loss.backward()
    optimizer.step()

    avg_episode_return = float(np.mean(episode_sums))
    return avg_episode_return

def train(env_name, seed, lr, num_iterations, batch_size, reward_to_go, advantage_norm, save_path):
    env = gym.make(env_name)
    set_seed(env, seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    avg_returns = []
    for it in range(1, num_iterations + 1):
        bs, ba, br = collect_episodes(env, policy, batch_size, gamma=0.99)
        avg_ret = policy_gradient_update(policy, optimizer, bs, ba, br,
                                         reward_to_go=reward_to_go,
                                         advantage_norm=advantage_norm,
                                         gamma=0.99)
        avg_returns.append(avg_ret)

        if it % max(1, num_iterations // 10) == 0:
            print(f"Iter {it}/{num_iterations}  AvgEpisodeReturn: {avg_ret:.2f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(policy.state_dict(), save_path)
        print("Saved policy to", save_path)

    env.close()
    return avg_returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--reward-to-go", action="store_true")
    parser.add_argument("--advantage-norm", action="store_true")
    parser.add_argument("--save", type=str, default="models/lunar_pg.pth")
    parser.add_argument("--no-save", dest="save_flag", action="store_true")
    args = parser.parse_args()

    save_path = None if args.save_flag else args.save
    returns = train(args.env, args.seed, args.lr, args.iterations, args.batch_size,
                    args.reward_to_go, args.advantage_norm, save_path)

    import matplotlib.pyplot as plt
    plt.plot(returns)
    plt.xlabel("Iteration")
    plt.ylabel("Average Episode Return")
    plt.title("Policy Gradient Training")
    plt.show()

if __name__ == "__main__":
    main()
