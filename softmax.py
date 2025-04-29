import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.random.randn(k)
    def pull(self, a):
        return np.random.randn() + self.means[a]

def run_softmax(bandit, steps=1000, tau=0.1):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = np.zeros(steps)
    for t in range(steps):
        # 計算機率分佈
        prefs = np.exp(Q / tau)
        probs = prefs / np.sum(prefs)
        a = np.random.choice(bandit.k, p=probs)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards[t] = r
    return rewards

def simulate_softmax(tau_list, runs=500, steps=1000):
    avg = {tau: np.zeros(steps) for tau in tau_list}
    for _ in range(runs):
        b = Bandit()
        for tau in tau_list:
            avg[tau] += run_softmax(b, steps, tau)
    for tau in tau_list:
        avg[tau] /= runs
    return avg

tau_list = [0.01, 0.1, 1.0]
avg_rewards = simulate_softmax(tau_list)
plt.figure()
for tau in tau_list:
    plt.plot(avg_rewards[tau], label=f'τ={tau}')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Softmax Performance Comparison')
plt.legend()
plt.show()