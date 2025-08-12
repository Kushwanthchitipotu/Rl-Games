# Auto-exported from notebook. Heuristic fixes applied for gymnasium API.
# Please review and run. If you want, I can add argparse and CLI flags.


import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, batch_size=64, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, env):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())




env_name = "MountainCar-v0"
lr = 0.001
gamma = 0.99
batch_size = 64
num_episodes = 5000
epsilon_decay = 0.995
min_epsilon = 0.01

env = gym.make(env_name)
agent = Agent(env.observation_space.shape[0], env.action_space.n, lr=lr, gamma=gamma, batch_size=batch_size, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

print("State space:", env.observation_space)
print("Action space:", env.action_space)




episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.choose_action(state, env)
        next_state, _, terminated, truncated, _ = env.step(action)
done = terminated or truncated

        position, velocity = next_state
        reward = (position - 0.5) + (velocity * 10)
        if position >= 0.5:
            reward += 100

        total_reward += reward
        agent.replay_buffer.append((state, action, reward, next_state, done))
        agent.train_step()
        state = next_state

    episode_rewards.append(total_reward)
    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    agent.update_target()

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

env.close()




n = 10
mean_rewards = [np.mean(episode_rewards[max(0, i - n):(i + 1)]) for i in range(len(episode_rewards))]

plt.plot(range(len(mean_rewards)), mean_rewards, label="Mean Reward")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.title("Learning Curve")
plt.show()




position_range = np.linspace(-1.2, 0.6, 50)
velocity_range = np.linspace(-0.07, 0.07, 50)

action_map = np.zeros((50, 50))

for i, position in enumerate(position_range):
    for j, velocity in enumerate(velocity_range):
        state = np.array([position, velocity])
        action = agent.choose_action(state, env)
        action_map[i, j] = action

plt.imshow(action_map, extent=[-0.07, 0.07, -1.2, 0.6], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(ticks=[0, 1, 2], label="Action (0: Left, 1: No Push, 2: Right)")
plt.xlabel("Velocity")
plt.ylabel("Position")
plt.title("Action Space Map")
plt.show()




def train_with_learning_rate(lr):
    agent = Agent(env.observation_space.shape[0], env.action_space.n, lr=lr, gamma=gamma, batch_size=batch_size, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
    rewards = []

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, env)
            next_state, _, terminated, truncated, _ = env.step(action)
done = terminated or truncated

            position, velocity = next_state
            reward = (position - 0.5) + (velocity * 10)
            if position >= 0.5:
                reward += 100

            total_reward += reward
            agent.replay_buffer.append((state, action, reward, next_state, done))
            agent.train_step()
            state = next_state

        rewards.append(total_reward)
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        agent.update_target()

    return rewards




learning_rates = [0.0005, 0.001, 0.005, 0.01]
learning_curves = {}

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    rewards = train_with_learning_rate(lr)
    learning_curves[lr] = [np.mean(rewards[max(0, i - n):(i + 1)]) for i in range(len(rewards))]

plt.figure(figsize=(10, 6))
for lr, rewards in learning_curves.items():
    plt.plot(rewards, label=f'LR = {lr}')

plt.xlabel("Episodes")
plt.ylabel("Mean Reward")
plt.legend()
plt.title("Learning Curve for Different Learning Rates")
plt.show()



