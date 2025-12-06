import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

data  = np.array([42, 57, 38, 63, 51, 46, 70])

mtbf = np.mean(data)
print("MTBF(mean time between failures): ", mtbf)

lambda_ = 1 / mtbf
print("lambda: ", lambda_)

x = np.linspace(0, 100, 1000)
y = lambda_ * np.exp(-lambda_ * x)

plt.plot(x, y)
plt.xlabel('时间/小时')
plt.ylabel('概率密度')
plt.title('pdf指数分布')
plt.hlines([i/500 for i in range(11)], 0, 100,linestyles=':',alpha=0.3,colors='gray')
plt.vlines([i for i in range(0,101,25)], 0, .02,linestyles=':',alpha=0.3,colors='gray')
plt.savefig(os.path.join(IMG_DIR, 'exponential_pdf.png'))
plt.show()

x = np.linspace(0, 200, 1000)
y = 1 - np.exp(-lambda_ * x)

plt.plot(x, y)
plt.xlabel('时间/小时')
plt.ylabel('累计概率密度')
plt.title('cdf指数分布')
plt.hlines([i/10 for i in range(11)], 0, 200,linestyles=':',alpha=0.3,colors='gray')
plt.vlines([i for i in range(0,201,25)], 0, 1,linestyles=':',alpha=0.3,colors='gray')
plt.savefig(os.path.join(IMG_DIR, 'exponential_cdf.png'))
plt.show()
