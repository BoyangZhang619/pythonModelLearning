import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from math import exp, factorial
from scipy.stats import poisson

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

np.random.seed(619)

# 参数
lambda_rate = 100  # 平均每分钟接100个电话
minutes = 600000

# 生成数据
calls_per_minute = np.random.poisson(lam=lambda_rate, size=minutes)
count = Counter(calls_per_minute)
vals = sorted(count.items(), key=lambda x: x[0])  # (电话数, 对应分钟数)

print("vals 示例（部分）:", vals[:10])
print("平均接到电话次数:", np.mean(calls_per_minute))
print("最大接到电话次数:", np.max(calls_per_minute))

# 方法一：用实际电话数作为 x（更直观）
xs = [v[0] for v in vals]
ys = [v[1] for v in vals]

plt.figure(figsize=(10,4))
plt.bar(xs, ys, align='center')
plt.xlabel('电话数量（每分钟）')
plt.ylabel('对应分钟数')
plt.title(f'每分钟接到的电话数量（泊松分布，λ={lambda_rate}，样本={minutes}）')
plt.xlim(min(xs)-0.5, max(xs)+0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "poisson_distribution_counts.png"))
plt.show()
plt.scatter(xs, [poisson.pmf(k, lambda_rate) for k in xs], linestyle='-', marker="o",s=5)
plt.title("理论泊松 pmf")
plt.xlim(min(xs)-0.5, max(xs)+0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "poisson_distribution_pmf.png"))
plt.show()
