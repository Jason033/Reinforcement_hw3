# Reinforcement_hw3

## Epsilon-Greedy

### 1. 演算法公式

![Epsilon-Greedy 公式](https://github.com/user-attachments/assets/b4beae89-a9e8-4ed5-9ea5-834325d36670)

其中，  
- \(Q_t(a)\) 為第 \(t\) 步估計的動作 \(a\) 的平均報酬  
- \(N_t(a)\) 為到時間 \(t\) 為止選擇動作 \(a\) 的次數  
- \(R_t\) 為執行動作 \(A_t\) 所得到的即時報酬  

---

### 2. ChatGPT Prompt

> **Prompt：**  
> "Please explain the Epsilon-Greedy algorithm for solving the Multi-Armed Bandit (MAB) problem. Include the mathematical formula for action selection, the role of epsilon (\(\epsilon\)), and how it balances exploration and exploitation. Provide examples showing how different epsilon values affect learning behavior. Additionally, summarize the main advantages and disadvantages of using Epsilon-Greedy."

---
### 3.程式碼與圖表
```python
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

```
![output](https://github.com/user-attachments/assets/9139b026-b2fa-4ecb-897d-c248763771ef)

### 4. 結果解釋（純文字格式）

#### ε 的角色

- ε（通常取 0.01–0.1）決定「探索」的機率：  
  - 以 ε 機率隨機選動作（探索）  
  - 以 ε 機率選估計報酬最高的動作（利用）  

- ε 越大：探索越多，能更全面嘗試各臂  
- ε 越小：較早收斂至目前估計最優臂，但易陷入次優  

#### 學習行為差異

- **ε = 0.01**  
  幾乎總是取報酬最高的臂，收斂快，但探索不足，易陷入次優解  

- **ε = 0.1**  
  適度探索與利用，能較早找到最優解，並維持一定程度的探索避免陷入局部最優  

- **ε = 0.2**  
  探索較多，初期報酬低，但可有效搜尋最佳臂；收斂較慢  

#### 優點

- 實作簡單，易於調參  
- 能在探索與利用間取得平衡，適用大多數情境  

#### 缺點

- 固定 ε 使探索率不可隨學習進度自動調整  
- 隨機探索可能浪費大量嘗試不佳動作的機會  
