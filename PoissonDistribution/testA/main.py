import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

lambda_ = 3.2
t = 5
mu = lambda_ * t
total = 30
for k in range(total):
    print(f"P(X={k}) = {poisson.pmf(k, mu)}")

y = poisson.pmf(np.arange(total), mu)
plt.bar(np.arange(total), y)
plt.xlabel('报警次数 k')
plt.ylabel('概率 P(X=k)')
plt.title(f'未来 {t} 分钟报警次数分布（泊松分布） λt={mu:.2f}')
plt.savefig(os.path.join(IMG_DIR, 'poisson.png'))
plt.show()