import numpy as np
import matplotlib.pyplot as plt

# 多臂賭徒環境
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.random.randn(k)
    def pull(self, a):
        return np.random.randn() + self.means[a]

# UCB 演算法
def run_ucb(bandit, steps=1000, alpha=2):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = np.zeros(steps)
    # 每個臂至少試一次
    for a in range(bandit.k):
        r = bandit.pull(a)
        Q[a], N[a], rewards[a] = r, 1, r
    for t in range(bandit.k, steps):
        ucb = Q + alpha * np.sqrt(2 * np.log(t+1) / N)
        a = np.argmax(ucb)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards[t] = r
    return rewards

# 平均化模擬
def simulate_ucb(alpha, runs=2000):
    steps = 1000
    avg = np.zeros(steps)
    for _ in range(runs):
        avg += run_ucb(Bandit(10), steps, alpha)
    return avg / runs

avg_rewards = simulate_ucb(alpha=2)
plt.plot(avg_rewards, label='UCB (α=2)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('UCB Algorithm Performance')
plt.legend()
plt.show()
