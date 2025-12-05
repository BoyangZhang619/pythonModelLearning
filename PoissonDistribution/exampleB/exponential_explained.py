import numpy as np

# ...existing code...
intervals = np.array([1.2, 2.1, 1.8, 2.3, 1.7, 2.0, 1.9, 2.5, 1.8, 2.2])

# 平均间隔
mean_interval = intervals.mean()

# λ = 每分钟顾客到达率
lambda_rate = 1 / mean_interval
print("到达率 λ =", lambda_rate)

# =============================
# 指数分布：下一位顾客来临概率（数值演示）
# =============================

# 计算生存函数 S(t) = P(T > t) = exp(-λ t)
s_1 = np.exp(-lambda_rate * 1)
s_3 = np.exp(-lambda_rate * 3)

# 计算在 t 分钟内至少来一位的概率 P(T <= t) = 1 - S(t)
p_within_1 = 1 - s_1
p_within_3 = 1 - s_3

print("S(1) = exp(-λ*1) =", s_1)
print("P(T ≤ 1) = 1 - exp(-λ*1) =", p_within_1)
print("S(3) = exp(-λ*3) =", s_3)
print("P(T ≤ 3) = 1 - exp(-λ*3) =", p_within_3)

# 验证概率边界：应在 [0,1]
assert 0 <= p_within_1 <= 1
assert 0 <= p_within_3 <= 1

# 简短结论说明（注释）
# - 使用 1 - exp(-λ t) 是因为我们想要“在 t 时间内至少发生一次到达”的概率，
#   它等于 1 减去“在 t 时间内没有到达”的概率（即生存函数）。
# - 虽然指数中有负号（-λ t），但 exp(负数) 是正数且在 (0,1]，所以 1 - exp(负数) 是一个合法的概率值。

