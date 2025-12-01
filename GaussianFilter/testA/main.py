import os

import numpy.random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

numpy.random.seed(619)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
EXCEL_DIR = os.path.join(BASE_DIR, "excel")

def load_data(file_path):
    data = pd.read_excel(os.path.join(EXCEL_DIR,file_path))
    seconds = np.array(data["帧号"] / 30)
    data = data.iloc[:, 1:]
    return data, seconds

def fix_zero_rows(data):
    cols = data.columns
    zero_frac = (data == 0.0).mean(axis=1)
    zero_indexes = data[zero_frac == 1].index

    for i in zero_indexes:
        if i == 0:
            data.loc[i, cols] = data.loc[i + 1, cols] - (data.loc[i + 2, cols] - data.loc[i + 1, cols])
            continue
        if i == len(data) - 1:
            data.loc[i, cols] = data.loc[i - 1, cols] - (data.loc[i - 2, cols] - data.loc[i - 1, cols])
            continue

        upper = data.loc[i - 1, cols]
        lower = data.loc[i + 1, cols]
        data.loc[i, cols] = (upper + lower) / 2

    return data

def detect_jump_frames(data, a=-2,b=-1.0,c=3,d=15):
    y = data["30_Y"].values

    vy = np.gradient(y)
    ay = np.gradient(vy)

    VEL_LIFT = -3.04
    ACC_LIFT = -1.4285
    MIN_AIR_FRAMES = 4

    # 起跳
    candidates = np.where((vy < VEL_LIFT) & (ay < ACC_LIFT))[0]
    jump_start = None
    for c in candidates:
        if c + MIN_AIR_FRAMES < len(y):
            if all(vy[c:c + MIN_AIR_FRAMES] < 0):
                jump_start = c
                break

    # 落地
    jump_end = None
    if jump_start is not None:
        baseline = np.mean(y[jump_start - 5:jump_start])
        threshold = baseline - 3

        for i in range(jump_start + 1, len(y)):
            if y[i] > threshold:
                if i + 3 < len(y):
                    if np.std(y[i:i + 3]) < 1.2:
                        jump_end = i
                        break
    jump_end = jump_end if jump_end is not None else 0
    jump_start = jump_start if jump_start is not None else 0
    return {
        "起跳帧": int(jump_start),
        "落地帧": int(jump_end),
    }

def showLine(data, points, start, end, message="自定义", isLegend=False):
    # 需要提取的列：points 中每个点的 x 列、y 列
    cols = [x for i in points for x in (i * 2, i * 2 + 1)]

    fig, axs = plt.subplots(2, 1, figsize=(15, 12))

    # 为避免覆盖，原始 data 不能动
    for ax, e in zip(axs, [0, 20]):
        # 每一轮都重新切片
        sliced = data.iloc[start - e : end + e, cols]

        data_x = sliced.iloc[:, ::2]
        data_y = sliced.iloc[:, 1::2]

        if isLegend:
            data_x.plot(ax=ax, label=[f"{p}_x" for p in points])
            data_y.plot(ax=ax, label=[f"{p}_y" for p in points])
            ax.legend()
        else:
            ax.plot(data_x.values)
            ax.plot(data_y.values)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.invert_yaxis()

        # 每个子图独立标题
        ax.set_title(f"{(start-e, end+e)} | {message}")
        ax.set_xlabel("帧数")
        ax.set_ylabel("相对距离/高度")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,'gaussian_filtered_data_lines.png'))
    plt.show()

if __name__ == "__main__":
    data,_ = load_data("运动者的跳远位置信息.xlsx")
    fix_zero_rows(data)
    data = data.apply(lambda col: gaussian_filter(col, sigma=2))
    result = detect_jump_frames(data)
    showLine(data, [30, 32], start=int(result["起跳帧"] - 20), end=int(result["落地帧"] + 20), message="运动者脚的跳远位置信息图示", isLegend=True)


