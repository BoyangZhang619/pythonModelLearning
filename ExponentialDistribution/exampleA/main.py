import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
from fractions import Fraction
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# 参数设置
lambda_rate = 1/2
scale = 1 / lambda_rate
num_events = 5000

# ------------------------------------------------------------
# 使用 scipy.stats.expon 的核心方法
# ------------------------------------------------------------
rv = expon(scale=scale)   # 指数分布对象

mean_val = rv.mean()
var_val = rv.var()
std_val = rv.std()
median_val = rv.median()

# PDF / CDF / SF / PPF
x = np.linspace(0, 40, 500)
pdf_vals = rv.pdf(x)
cdf_vals = rv.cdf(x)
sf_vals = rv.sf(x)
ppf_vals = rv.ppf(np.linspace(0, 1, 500))

# ------------------------------------------------------------
# 采样
intervals = rv.rvs(size=num_events)
# ------------------------------------------------------------

# 输出数值
print("样本均值:", round(np.mean(intervals),3))
print("理论均值:", mean_val)
print("样本方差:", round(np.var(intervals),3))
print("理论方差:", var_val)
print("样本标准差:",round(np.std(intervals),3))
print("理论标准差:", std_val)
print("样本中位数:",round(np.median(intervals),3))
print("理论中位数:", median_val)
print("最短间隔:", round(np.min(intervals),6))
print("最长间隔:", round(np.max(intervals),6))

# ============================================================
# 图表 1：样本直方图（原图）
# ============================================================
plt.figure(figsize=(10, 4))
plt.hist(intervals, bins=100, edgecolor='black')
plt.xlabel('时间间隔（分钟）')
plt.ylabel('事件次数')
plt.title('样本间隔时间分布（指数分布）')
plt.savefig(os.path.join(IMG_DIR, "hist_intervals.png"))
plt.show()

# ============================================================
# 图表 2：PDF(probability density function) 曲线
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(x, pdf_vals)
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('指数分布 PDF')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(IMG_DIR, "pdf.png"))
plt.show()

# ============================================================
# 图表 3：CDF(cumulative distribution function) 曲线
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(x, cdf_vals)
plt.xlabel('x')
plt.ylabel('CDF')
plt.title('指数分布 CDF')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(IMG_DIR, "cdf.png"))
plt.show()

# ============================================================
# 图表 4：SF(survival function) 生存函数
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(x, sf_vals)
plt.xlabel('x')
plt.ylabel('SF')
plt.title('指数分布 生存函数')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(IMG_DIR, "sf.png"))
plt.show()

# ============================================================
# 图表 5：PPF(percent point function)（分位数函数）
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, 1, 500), ppf_vals)
plt.xlabel('概率 q')
plt.ylabel('PPF')
plt.title('指数分布 PPF（分位数函数）')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(IMG_DIR, "ppf.png"))
plt.show()
