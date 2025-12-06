# Simulate data and demonstrate a composite solution that uses:
# - Poisson distribution (counts per interval)
# - Exponential distribution (inter-arrival times)
# - Gaussian filter (smoothing observed speed time series)
# - Kalman Filter (approximate, for estimating the latent Poisson rate)
#
# The scenario: 道路检测器每分钟统计到的车辆数（counts）。真实到达率 λ_t 随时间缓慢变化。
# 我们用 Poisson 生成 counts；用 exponential 检验/生成到达间隔；用 Gaussian filter 平滑速度观测；
# 用（近似）Kalman filter 来估计 λ_t（将 Poisson 观测近似为 带方差 = λ 的高斯观测）。
#
# 输出：图像（真实 λ、估计 λ、counts、平滑速度）和关键指标（RMSE, MLE rate）以及示例数据表。
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy.stats import expon

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'img')
CSV_DIR = os.path.join(BASE_DIR, 'csv')

np.random.seed(42)

# simulate
T = 300  # time steps (minutes)
t = np.arange(T)

# true underlying rate λ_t: baseline + slow sinusoidal + random walk
baseline = 5.0
amp = 3.0
season = 2 * np.pi * t / 120.0  # period ~120
lambda_true = baseline + amp * (np.sin(season)) + 0.3 * np.cumsum(np.random.normal(scale=0.1, size=T))
lambda_true = np.clip(lambda_true, 0.1, None)  # avoid nonpositive

# generate counts per minute as Poisson(lambda_true)
counts = np.random.poisson(lam=lambda_true)

# generate inter-arrival times by assuming within each minute the interarrivals are expo(rate=lambda_true)
interarrival_times = []
for i in range(T):
    lam = lambda_true[i]
    k = counts[i]
    # sample k interarrival times if k>0 from exponential(lam); these are within-minute intervals (relative)
    if k > 0:
        iats = np.random.exponential(1.0/lam, size=k)
        interarrival_times.extend(iats.tolist())

interarrival_times = np.array(interarrival_times) if len(interarrival_times)>0 else np.array([])

# simulate average speed per minute: speed decreases with increasing λ (congestion effect)
v_free = 60.0  # free-flow speed km/h
alpha = 2.0    # how strongly rate affects speed
speed_noise_std = 3.0
speed_obs = v_free - alpha * lambda_true + np.random.normal(scale=speed_noise_std, size=T)

# apply Gaussian smoothing filter to speed_obs (1D gaussian kernel convolution)
def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = int(ceil(3*sigma))
    x = np.arange(-radius, radius+1)
    k = np.exp(-0.5*(x/sigma)**2)
    k = k / k.sum()
    return k

sigma = 2.5  # smoothing in minutes
kernel = gaussian_kernel1d(sigma)
speed_smooth = np.convolve(speed_obs, kernel, mode='same')

# Approximate Kalman filter to estimate lambda_t from counts.
# State: x_t = lambda_t (we attempt to track it). Transition: x_{t+1} = x_t + process_noise (random walk).
# Observation: y_t = counts_t ~ Poisson(x_t). Approximate Poisson as Gaussian with mean x_t and var x_t.
Q = 0.2  # process noise variance (tuneable)
x_est = np.zeros(T)
P = np.zeros(T)
# init
x_prev = max(1.0, np.mean(counts[:10]))  # initial guess
P_prev = 1.0

for i in range(T):
    # prediction
    x_pred = x_prev
    P_pred = P_prev + Q

    # observation variance approx = Poisson variance = x_pred (but ensure >= small eps)
    R = max(0.5, x_pred)

    # Kalman gain
    K = P_pred / (P_pred + R)

    # update with observed count
    y = counts[i]
    x_upd = x_pred + K * (y - x_pred)
    P_upd = (1 - K) * P_pred

    x_est[i] = x_upd
    P[i] = P_upd

    x_prev = x_upd
    P_prev = P_upd

# Fit exponential MLE to interarrival_times (rate = 1/mean)
if interarrival_times.size > 0:
    mle_rate = 1.0 / np.mean(interarrival_times)
else:
    mle_rate = np.nan

# Metrics
rmse_lambda = np.sqrt(np.mean((x_est - lambda_true)**2))
mae_lambda = np.mean(np.abs(x_est - lambda_true))

# Prepare a small DataFrame with example rows
df = pd.DataFrame({
    't_min': t,
    'lambda_true': lambda_true,
    'counts': counts,
    'speed_obs': speed_obs,
    'speed_smooth': speed_smooth,
    'lambda_est_kalman': x_est
})

# Display metrics (use a DataFrame for neat UI)
metrics = pd.DataFrame({
    'metric': ['RMSE_lambda', 'MAE_lambda', 'MLE_rate_from_interarrival', 'total_events'],
    'value': [rmse_lambda, mae_lambda, mle_rate, int(counts.sum())]
})

# Plot 1: true lambda vs estimated
plt.figure(figsize=(10,4))
plt.plot(t, lambda_true)
plt.plot(t, x_est)
plt.title('真实 λ (True) 与 卡尔曼估计 λ_est (Estimated)')
plt.xlabel('时间 (分钟)')
plt.ylabel('λ (车辆/分)')
plt.legend(['lambda_true','lambda_est_kalman'])
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'真实λ与卡尔曼估计λ_est对比.png'))
plt.show()

# Plot 2: counts time series
plt.figure(figsize=(10,3))
plt.bar(t, counts)
plt.title('每分钟观测到的车辆数 (counts)')
plt.xlabel('时间 (分钟)')
plt.ylabel('counts')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'每分钟counts(柱状图).png'))
plt.show()

# Plot 3: observed speed and Gaussian-smoothed speed
plt.figure(figsize=(10,3.5))
plt.plot(t, speed_obs)
plt.plot(t, speed_smooth)
plt.title('观测速度与高斯滤波平滑结果 (speed_obs vs speed_smooth)')
plt.xlabel('时间 (分钟)')
plt.ylabel('速度 (km/h)')
plt.legend(['speed_obs','speed_smooth'])
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,'观测速度与高斯滤波平滑速度对比.png'))
plt.show()

print(df.sample(20))
print(metrics)

# Save simulated dataset to /mnt/data for user download
df.to_csv(os.path.join(CSV_DIR,'simulated_traffic_data.csv'), index=False)
metrics.to_csv(os.path.join(CSV_DIR,'simulated_metrics.csv'), index=False)
print("[文件已保存] /csv/simulated_traffic_data.csv,/csv/ simulated_metrics.csv")
