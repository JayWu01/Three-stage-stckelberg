import matplotlib.pyplot as plt
import numpy as np

# 创建一个2x2的网格
fig, axs = plt.subplots(3, 3)

# 在每个子图中绘制一个图形
for ax in axs.flat:
    # 随机生成一些数据
    x = np.random.rand(100)
    y = np.random.rand(100)
    # 绘制散点图
    ax.scatter(x, y)

# 显示图形
plt.show()