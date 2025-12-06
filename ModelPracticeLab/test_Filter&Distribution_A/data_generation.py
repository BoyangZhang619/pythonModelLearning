# generation code mainly from gpt5
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, 'csv')

np.random.seed(2025)

# 时间长度（秒）
T = 600

t = np.arange(T)

# 基础请求强度
base_rate = 8

# 慢变化
trend = 2 * np.sin(2 * np.pi * t / 200)

# 突发 burst
burst = np.zeros(T)
burst[200:240] += 10
burst[420:450] += 15

lambda_t = np.clip(base_rate + trend + burst, 0.1, None)

# 每秒请求数（Poisson）
request_count = np.random.poisson(lambda_t)

# 请求到达间隔（Exponential）
inter_arrival_times = []
inter_arrival_second = []

for sec in range(T):
    n = request_count[sec]
    if n > 0:
        iats = np.random.exponential(scale=1 / lambda_t[sec], size=n)
        inter_arrival_times.extend(iats)
        inter_arrival_second.extend([sec] * n)

inter_arrival_times = np.array(inter_arrival_times)

# 整理成 DataFrame
df_counts = pd.DataFrame({
    "second": t,
    "lambda_true": lambda_t,
    "request_count": request_count
})

df_iat = pd.DataFrame({
    "second": inter_arrival_second,
    "inter_arrival_time": inter_arrival_times
})

# 保存
df_counts.to_csv(os.path.join(CSV_DIR,"server_request_counts.csv"), index=False)
df_iat.to_csv(os.path.join(CSV_DIR,"server_inter_arrival_times.csv"), index=False)

print("数据已生成：")
print("/csv/server_request_counts.csv")
print("/csv/server_inter_arrival_times.csv")
