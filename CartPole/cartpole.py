# Auto-exported from notebook. Heuristic fixes applied for gymnasium API.
# Please review and run. If you want, I can add argparse and CLI flags.

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




env = gym.make('CartPole-v0')
print("State Space:", env.observation_space)
print("Action Space:", env.action_space)

done, state = False, env.reset()[0]
total_reward = 0
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
done = terminated or truncated
    total_reward += reward

print("Total reward obtained by random agent:", total_reward)




class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)




def compute_returns(rewards, gamma, reward_to_go):
    returns = []
    if reward_to_go:
        for t in range(len(rewards)):
            Gt = sum([gamma ** (k - t) * rewards[k] for k in range(t, len(rewards))])
            returns.append(Gt)
    else:
        G = sum([gamma ** t * r for t, r in enumerate(rewards)])
        returns = [G] * len(rewards)
    return returns
def normalize_advantages(advantages):
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages
def policy_gradient(env, policy, optimizer, gamma=0.99, reward_to_go=True, advantage_norm=True, num_iterations=1000, batch_size=500):
    average_returns = []

    for i in range(num_iterations):
        states, actions, advantages = [], [], []
        episode_rewards = []
        episode_returns = []
        state, _ = env.reset()
        done = False

        while len(states) < batch_size:

            action_probs = policy(torch.tensor(state, dtype=torch.float32))
            action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
done = terminated or truncated

            states.append(state)
            actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            if done:

                returns = compute_returns(episode_rewards, gamma, reward_to_go)
                baseline = np.mean(returns)
                episode_advantages = returns - baseline if advantage_norm else returns
                if advantage_norm:
                    episode_advantages = normalize_advantages(episode_advantages)

                advantages.extend(episode_advantages)
                episode_returns.append(sum(episode_rewards))

                episode_rewards = []
                state, _ = env.reset()
                done = False


        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions[:len(advantages)], dtype=torch.int64)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)


        optimizer.zero_grad()
        log_probs = torch.log(policy(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze())
        loss = -torch.mean(log_probs * advantages_tensor)
        loss.backward()
        optimizer.step()


        avg_return = np.mean(episode_returns)
        average_returns.append(avg_return)
        print(f"Iteration {i + 1}/{num_iterations}, Average Return: {avg_return}")

    return average_returns




env = gym.make("CartPole-v0")
policy = PolicyNetwork(env.observation_space.shape[0], hidden_dim=128, output_dim=env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

average_returns_no_rtgo_no_norm = policy_gradient(env, policy, optimizer, reward_to_go=False, advantage_norm=False, num_iterations=5000)
average_returns_no_rtgo_norm = policy_gradient(env, policy, optimizer, reward_to_go=False, advantage_norm=True, num_iterations=5000)
average_returns_rtgo_no_norm = policy_gradient(env, policy, optimizer, reward_to_go=True, advantage_norm=False, num_iterations=5000)
average_returns_rtgo_norm = policy_gradient(env, policy, optimizer, reward_to_go=True, advantage_norm=True, num_iterations=5000)




plt.plot(average_returns_no_rtgo_no_norm, label="No Reward-to-Go, No Normalization")
plt.plot(average_returns_no_rtgo_norm, label="No Reward-to-Go & Normalization")
plt.plot(average_returns_rtgo_no_norm, label="Reward-to-Go, No Normalization")
plt.plot(average_returns_rtgo_norm, label="Reward-to-Go & Normalization")
plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.legend()
plt.title("Policy Gradient with Variance Reduction Techniques")
plt.show()




batch_sizes = [200, 300, 400,500]
all_returns = {}

for batch_size in batch_sizes:
    print(f"\nRunning policy gradient with batch size {batch_size}")
    returns = policy_gradient(env, policy, optimizer, reward_to_go=True, advantage_norm=True, num_iterations=1000, batch_size=batch_size)
    all_returns[batch_size] = returns






for batch_size, returns in all_returns.items():
    plt.plot(returns, label=f"Batch Size {batch_size}")

plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.legend()
plt.title("Impact of Batch Size on Policy Gradient Performance")
plt.show()


