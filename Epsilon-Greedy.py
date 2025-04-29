import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon, alpha=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.q_values[arm] += self.alpha * (reward - self.q_values[arm])

def simulate(epsilon, steps=500, true_rewards=None):
    n_arms = len(true_rewards)
    agent = EpsilonGreedy(n_arms, epsilon)
    rewards = []

    for step in range(steps):
        action = agent.select_action()
        reward = np.random.randn() + true_rewards[action]  # stochastic reward
        agent.update(action, reward)
        rewards.append(reward)
    return np.cumsum(rewards) / (np.arange(1, steps + 1))

# True rewards for each arm
true_rewards = [1.0, 1.5, 2.0, 1.2]

epsilons = [0.01, 0.1, 0.3]
steps = 500
results = {}

for eps in epsilons:
    avg_rewards = simulate(eps, steps, true_rewards)
    results[eps] = avg_rewards

# 繪製結果
plt.figure(figsize=(10, 6))
for eps, rewards in results.items():
    plt.plot(rewards, label=f'epsilon={eps}')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Epsilon-Greedy: Different Epsilon Values')
plt.legend()
plt.grid(True)
plt.show()
