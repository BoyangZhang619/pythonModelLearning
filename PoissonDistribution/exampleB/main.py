import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, poisson

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =============================
# 1. 历史间隔数据
# =============================
intervals = np.array([1.2, 2.1, 1.8, 2.3, 1.7, 2.0, 1.9, 2.5, 1.8, 2.2])

# 平均间隔
mean_interval = intervals.mean()

# λ = 每分钟顾客到达率
lambda_rate = 1 / mean_interval
print("到达率 λ =", lambda_rate)


# =============================
# 2. 指数分布：下一位顾客来临概率，求的是cdf
# =============================
p_within_1 = 1 - np.exp(-lambda_rate * 1)
p_within_3 = 1 - np.exp(-lambda_rate * 3)

print("下一分钟至少来一位顾客的概率 =", p_within_1)
print("3 分钟内至少来一位顾客的概率 =", p_within_3)


# =============================
# 3. 模拟未来到达间隔（指数分布）
# =============================
future_intervals = expon(scale=1/lambda_rate).rvs(size=5000)  #  按照指数分布采样
print("模拟未来间隔平均值 =", future_intervals.mean())


# =============================
# 4. 泊松分布：未来10分钟来 k 位顾客
# =============================
t = 10  # 未来 10 分钟
mu = lambda_rate * t  # 泊松分布参数

print("\n未来 10 分钟来顾客次数的泊松概率：")
for k in range(11):
    print(f"P(X={k}) =", poisson.pmf(k, mu))


# =============================
# 5. 图 1：历史间隔 + 指数分布拟合
# =============================
plt.figure(figsize=(12, 4))

# 直方图（历史间隔）
plt.hist(intervals, bins=6, density=True, alpha=0.6, label="历史间隔分布")

# 拟合指数分布曲线
x = np.linspace(0, max(intervals) + 1, 200)
pdf = expon(scale=1/lambda_rate).pdf(x)
plt.plot(x, pdf, linewidth=2, label=f"指数分布拟合 (λ={lambda_rate:.2f})")

plt.title("顾客到达间隔分布 & 指数分布拟合")
plt.xlabel("间隔（分钟）")
plt.ylabel("概率密度")
plt.legend()


# =============================
# 6. 图 2：未来10分钟内顾客次数的泊松分布
# =============================
plt.figure(figsize=(12, 4))

ks = np.arange(0, 15)
pmf_vals = poisson.pmf(ks, mu) #  probability mass function
print(f"Total probability:{sum(pmf_vals)}")
plt.bar(ks, pmf_vals, alpha=0.7)
plt.title(f"未来 {t} 分钟顾客到达次数分布（泊松分布） λt={mu:.2f}")
plt.xlabel("顾客数 k")
plt.ylabel("概率 P(X=k)")

plt.tight_layout()
plt.show()
