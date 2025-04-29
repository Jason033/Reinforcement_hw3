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

### 4. 結果解釋

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


## UCB (Upper Confidence Bound)

### 1. 演算法公式

![image](https://github.com/user-attachments/assets/c5d87fe4-cbc4-47f7-94ab-7dc3b76082c0)

### 2. ChatGPT Prompt

Please explain the UCB (Upper Confidence Bound) algorithm for the Multi-Armed Bandit problem. 
Derive the mathematical formula for the UCB score calculation, clearly explain each term 
(mean reward, confidence bound), and describe how it naturally balances exploration and 
exploitation. Provide an example of UCB selection over time and discuss its advantages 
compared to Epsilon-Greedy. Also mention any limitations UCB may have in practice.

### 3.程式碼與圖表

```python
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.random.randn(k)
    def pull(self, a):
        return np.random.randn() + self.means[a]

def run_ucb(bandit, steps=1000, c=2):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = np.zeros(steps)
    # 初始化，每隻手臂先拉一次
    for a in range(bandit.k):
        r = bandit.pull(a)
        Q[a], N[a], rewards[a] = r, 1, r
    for t in range(bandit.k, steps):
        ucb = Q + c * np.sqrt(np.log(t + 1) / N)
        a = np.argmax(ucb)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards[t] = r
    return rewards

def simulate_ucb(c_values, runs=500, steps=1000):
    avg = {c: np.zeros(steps) for c in c_values}
    for _ in range(runs):
        b = Bandit()
        for c in c_values:
            avg[c] += run_ucb(b, steps, c)
    for c in c_values:
        avg[c] /= runs
    return avg

c_list = [0.5, 1, 2]
avg_rewards = simulate_ucb(c_list)
plt.figure()
for c in c_list:
    plt.plot(avg_rewards[c], label=f'c={c}')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('UCB1 Performance Comparison')
plt.legend()
plt.show()
```
![output (1)](https://github.com/user-attachments/assets/46567132-3b86-4d60-8d4b-9dd93509068e)

### 4. 結果解釋

- **探索參數 `c` 的影響**  
  - `c` 值越大：置信界（confidence bound）項越大，偏重探索；初期波動較高  
  - `c` 值越小：偏重利用，容易過早收斂到局部最優

- **自然平衡探索與利用**  
  - 結合平均報酬估計與置信界，對不常選擇的臂給予額外獎勵  
  - 隨著 `N_t(a)` 增加，置信界收縮，逐漸偏向平均報酬更高的臂

- **與 Epsilon-Greedy 的比較**  
  - **優點**  
    - 自動調整探索強度，無需預先設定固定的 ε  
    - 理論保證：持續探索次優臂，降低漏掉最佳臂的風險  
  - **缺點**  
    - 計算開銷較大：需持續計算對數與開根號  
    - 非平穩環境下適應不佳：臂的報酬分佈若隨時間改變，性能下降

- **實務限制**  
  - 初期需對每隻臂至少拉一次，若臂數量龐大，開銷不可忽略  
  - 計算 UCB 值的額外成本，對高維臂數場景挑戰較大

## Softmax (Action Selection)

### 1. 演算法公式
![image](https://github.com/user-attachments/assets/178f6d78-666b-499c-93ff-4eef0976b754)



### 2. ChatGPT Prompt

```text
"Please explain the Softmax action selection algorithm in the context of the 
Multi-Armed Bandit problem. Include the mathematical formula for computing the 
probability of choosing each arm based on estimated rewards and a temperature 
parameter τ. Explain how different temperature values affect exploration versus 
exploitation. Provide example scenarios and summarize the strengths and weaknesses 
of Softmax compared to Epsilon-Greedy and UCB."
```

### 3. 程式碼與圖表

```python
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
```

![image](https://github.com/user-attachments/assets/8f1c0ac8-9496-47d3-8aca-13d43ee4bbee)


### 4. 結果解釋

- **溫度參數 `τ` 的影響**  
  - `τ` 越小：分佈越集中，偏重利用；探索機率降低  
  - `τ` 越大：分佈越平坦，偏重探索；各臂被選中機率更接近

- **探索與利用平衡**  
  - Softmax 根據相對估計值自適應分配探索機率  
  - 不像 ε-Greedy 固定機率隨機，與 UCB 的置信界計算也不同

- **與 Epsilon-Greedy 的比較**  
  - **優點**  
    - 根據估計值軟分配機率，細緻地平衡探索  
    - 無需「隨機全臂」探索，效率更高  
  - **缺點**  
    - 需計算指數與機率分佈，運算量較大  
    - 對 `τ` 敏感，需妥善調參

- **與 UCB 的比較**  
  - **優點**  
    - 機率選擇更平滑，避免過度依賴置信界  
    - 同樣無需固定 ε  
  - **缺點**  
    - 無理論上界（regret bound）保證  
    - 在極低 `τ` 或極高 `τ` 時，行為可能退化

- **實務建議**  
  - 可採用動態調整 `τ`（如隨時間衰減）以兼顧初期探索與後期利用  
  - 選擇算法時，需考慮環境非平穩性與計算成本因素

## Thompson Sampling

### 1. 演算法公式

![image](https://github.com/user-attachments/assets/37c10936-eff3-4b19-bada-3db9e8ac0db9)


### 2. ChatGPT Prompt

```text
"Please explain the Thompson Sampling algorithm for solving the Multi-Armed 
Bandit problem. Describe the Bayesian intuition behind it, the posterior 
updating process, and the arm selection rule. Provide the corresponding 
mathematical formulas for Beta distribution updates in the Bernoulli reward case. 
Discuss the practical advantages and possible limitations of Thompson Sampling 
compared to other methods like UCB and Epsilon-Greedy."
```

### 3. 程式碼與圖表

```python
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
```

![image](https://github.com/user-attachments/assets/79f13561-061e-4892-8649-b4e6d1792c3d)


### 4. 結果解釋

- **貝葉斯直覺**  
  - 以 Beta 分布作為對每個臂成功機率的先驗，隨資料累積動態更新後驗  
  - 透過從後驗分佈抽樣 \(\theta_t(a)\)，在未知臂上保有探索機會

- **後驗更新過程**  
  - 成功 \((R_t=1)\) 時：\(\alpha \leftarrow \alpha + 1\)  
  - 失敗 \((R_t=0)\) 時：\(\beta \leftarrow \beta + 1\)

- **臂選擇規則**  
  - 對每個臂抽樣 \(\theta\)，選擇擁有最大 \(\theta\) 的臂，自然平衡探索與利用

- **優點**  
  - 無需調整探索參數，依後驗不確定性自動決定探索強度  
  - 在實務中常能達到低後悔值 (regret)

- **缺點**  
  - 每步需對所有臂抽樣並計算 Beta 分布，計算開銷較大  
  - 需要對 reward 模型假設合理，對非伯努利獎勵需改用其他後驗分布

- **與其他方法比較**  
  - 相對 Epsilon-Greedy：能更精細地依不確定性進行探索  
  - 相對 UCB：不需動態維護置信界計算，但理論後悔界較難推導  
