import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, k=10):
        self.k = k
        self.probs = np.random.rand(k)
    def pull(self, a):
        return np.random.rand() < self.probs[a]

def run_thompson(bandit, steps=1000):
    k = bandit.k
    alpha = np.ones(k)
    beta = np.ones(k)
    rewards = np.zeros(steps)
    for t in range(steps):
        theta = np.random.beta(alpha, beta)
        a = np.argmax(theta)
        r = bandit.pull(a)
        alpha[a] += r
        beta[a] += 1 - r
        rewards[t] = r
    return rewards

def simulate_thompson(runs=500, steps=1000):
    avg_rewards = np.zeros(steps)
    for _ in range(runs):
        b = BernoulliBandit()
        avg_rewards += run_thompson(b, steps)
    return avg_rewards / runs

avg_rewards = simulate_thompson()
plt.figure()
plt.plot(avg_rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Thompson Sampling Performance')
plt.show()