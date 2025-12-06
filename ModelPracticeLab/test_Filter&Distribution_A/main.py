import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子
np.random.seed(619)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')
CSV_DIR = os.path.join(BASE_DIR, 'csv')

# 读取数据
counts_df = pd.read_csv(os.path.join(CSV_DIR, 'server_request_counts.csv'))
iats_df = pd.read_csv(os.path.join(CSV_DIR, 'server_inter_arrival_times.csv'))

# ============================================================
# ✅ Task A：数据理解与建模
# 从数据中说明：
# 哪些量是 Poisson 随机变量
# 哪些量对应 Exponential 分布
# 用统计量或图形验证 Poisson ⇔ Exponential 的一致性关系
# ✅ Task B：参数估计
# 对请求间隔数据：
# 估计 Exponential 分布的 rate
# 对每秒请求数：
# 给出合理的 λ 估计策略
# ============================================================

request_count_mean = counts_df['request_count'].mean()
inter_arrival_time_mean_count_backwards = 1 / iats_df['inter_arrival_time'].mean()
print(f"单位时间平均请求次数:     {request_count_mean:.4f}")
print(f"各请求间平均间隔时间的倒数:{inter_arrival_time_mean_count_backwards:.4f}")
print(f"两者比值:{inter_arrival_time_mean_count_backwards/request_count_mean:.4f}(应接近1)")
print("这是 Poisson–Exponential 假设的必要但不充分条件。")

# 各个请求间的请求间隔时间的分布图(Exponential分布)
plt.figure(figsize=(10, 5))
plt.hist(iats_df['inter_arrival_time'],bins=200, label='连续样本间间隔时间')
plt.title('连续样本间间隔时间分布')
plt.xlabel('间隔时间')
plt.ylabel('次数')
plt.legend()
plt.savefig(os.path.join(IMG_DIR, 'inter_arrival_time.png'))
plt.show()

# 单位时间内的请求数的分布图(Poisson分布)
plt.figure(figsize=(10, 5))
plt.hist(counts_df['request_count'],bins=37, label='请求次数',alpha=0.5)
plt.title('单位时间请求次数对应频次分布')
plt.xlabel('单位时间请求次数')
plt.ylabel('频次')
plt.legend()
plt.savefig(os.path.join(IMG_DIR, 'request_count.png'))
plt.show()

# λ 估计
print(f"Exponential分布的rate:{1 / request_count_mean}")

# # # # # # lambda_hat_exp = 1 / iats_df['inter_arrival_time'].mean()
# # # # # #
# # # # # # plt.figure(figsize=(6, 6))
# # # # # # stats.probplot(
# # # # # #     iats_df['inter_arrival_time'],
# # # # # #     dist=stats.expon,
# # # # # #     sparams=(0, 1 / lambda_hat_exp),
# # # # # #     plot=plt
# # # # # # )
# # # # # # plt.title("QQ-plot: Inter-arrival Time vs Exponential")
# # # # # # plt.show()
# # # # # #
# # # # # # ks_stat, p_value = stats.kstest(
# # # # # #     iats_df['inter_arrival_time'],
# # # # # #     'expon',
# # # # # #     args=(0, 1 / lambda_hat_exp)
# # # # # # )
# # # # # #
# # # # # # print(f"KS statistic: {ks_stat:.4f}")
# # # # # # print(f"p-value: {p_value:.4f}")
# # # # # #这段是gpt后给的，我暂时还看不懂，所以注释了，嘻嘻

# ============================================================
# ✅ Task C：Gaussian 滤波
# 对原始请求数时间序列：
# 使用 Gaussian Filter 做平滑
# 解释平滑前后信号的差异
# 分析 burst 在平滑信号中的表现
# ============================================================

# 滤波前后各秒对应请求数表图
gaussian_request_count = gaussian_filter1d(counts_df['request_count'],sigma=10)
plt.figure(figsize=(10, 5))
plt.scatter(range(600),counts_df['request_count'],label='高斯滤波前的数据',alpha=.5,color='r',s=5)
plt.scatter(range(600),gaussian_request_count,label='高斯滤波后的数据',alpha=.5,color='b',s=5)
plt.title('滤波前后各秒对应请求数表图')
plt.xlabel('秒')
plt.ylabel('次数')
plt.legend()
plt.savefig(os.path.join(IMG_DIR, 'gaussian_request_count.png'))
plt.show()
# 平滑后的值更加连续且缓和
# 从图中可见在数据异常时有极明显的波出现
# Gaussian 滤波会削弱噪声，但不会“消灭”持续性的异常（burst）


