import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
def singleDimensionLine():
    data = np.random.normal(size=(1000, 1))
    # 应用高斯滤波
    gaussian_filtered_data = gaussian_filter(data, sigma=8)

    # 可视化原始数据和滤波后的数据
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title('Original Data')

    plt.subplot(2, 1, 2)
    plt.title('Gaussian Filtered Data')
    plt.plot(gaussian_filtered_data)
    plt.savefig(os.path.join(IMG_DIR,'gaussian_filtered_data_line.png'))
    plt.show()

def twoDimensionsGreyImage():
    # 生成二维随机小数 0~1
    data = np.random.rand(50, 50)  # 每个点都是 0-1 之间的小数

    # 高斯滤波
    gaussian_filtered_data = gaussian_filter(data, sigma=3)
    gaussian_filtered_data = gaussian_filter(gaussian_filtered_data, sigma=3)

    gaussian_filtered_data = gaussian_filtered_data ** 2
    # 可视化（直接当成灰度图，0=#000, 1=#fff）
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Random Float Data')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_filtered_data, cmap='gray', vmin=0, vmax=1)
    plt.title('Gaussian Filtered Data')
    plt.axis('off')
    plt.savefig(os.path.join(IMG_DIR,'gaussian_filtered_data_gray.png'))
    plt.show()



def twoDimensionColorfulImage():
    # 生成示例数据
    data = np.random.normal(size=(100, 100))
    # 应用高斯滤波
    gaussian_filtered_data = gaussian_filter(data, sigma=5)

    # 可视化原始数据和滤波后的数据
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(data, cmap='viridis')
    plt.title('Original Data')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_filtered_data, cmap='viridis')
    plt.title('Gaussian Filtered Data')
    plt.axis('off')
    plt.savefig(os.path.join(IMG_DIR,'gaussian_filtered_data_colorful.png'))
    plt.show()

if __name__ == "__main__":
    singleDimensionLine()
    twoDimensionsGreyImage()
    twoDimensionColorfulImage()