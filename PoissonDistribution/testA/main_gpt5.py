from scipy.stats import poisson
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

lambda_ = 3.2
t = 5
mu = lambda_ * t  # =16

# 建议的上界阈值（不同置信水平）
alpha = 0.10  # 90% 中心区间
lo90 = int(poisson.ppf(alpha / 2, mu))
hi90 = int(poisson.ppf(1 - alpha / 2, mu))

alpha2 = 0.05  # 95% 中心区间
lo95 = int(poisson.ppf(alpha2 / 2, mu))
hi95 = int(poisson.ppf(1 - alpha2 / 2, mu))

# 或者只取上界作为报警阈值
alarm_threshold = int(poisson.ppf(0.98, mu))  # 98% 上界

print("mu =", mu)
print("90% 区间:", lo90, hi90)
print("95% 区间:", lo95, hi95)
print("建议 98% 报警阈值:", alarm_threshold)

# 绘图范围：覆盖到 mu + 4*sqrt(mu) 足够（或 hi95 + 若干）
import math
max_k = max( int(mu + 6 * math.sqrt(mu)), hi95 + 5 )

y = poisson.pmf(np.arange(max_k + 1), mu)

plt.figure(figsize=(10,5))
bars = plt.bar(np.arange(max_k + 1), y, color='lightgray', edgecolor='k')

# 高亮 90% 区间
for k in range(lo90, hi90 + 1):
    bars[k].set_color('C0')  # 蓝色

# 高亮 95% 区间（更深色覆盖）
for k in range(lo95, hi95 + 1):
    bars[k].set_color('C1')  # 橙色

# 标出报警阈值线
plt.axvline(alarm_threshold, color='r', linestyle='--', label=f'95% 上界 = {alarm_threshold}')

plt.xlabel('报警次数 k')
plt.ylabel('概率 P(X=k)')
plt.title(f'未来 {t} 分钟报警次数分布（Poisson，mu={mu:.2f}）')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'poisson_with_thresholds.png'))
plt.show()
