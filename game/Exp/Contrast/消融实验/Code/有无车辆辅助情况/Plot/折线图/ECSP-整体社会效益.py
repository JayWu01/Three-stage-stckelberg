import matplotlib.pyplot as plt

# 预设字体格式，并传给rc方法
import numpy as np

# 四组数据
np.set_printoptions(precision=16)

socialWelfare = [[104.16466780433962, 101.63910423705691, 100.12796887802119, 98.88994072144799, 104.66186426440885,
                 105.34802425243409, 110.86269554887856, 106.78718981449744, 95.64683665570371],
                 [104.97567099862871, 106.93431552683454, 111.21938868946728, 114.64539331691222, 117.43557578889813,
                 119.85208663898406, 124.78170655479589, 125.83472284579223, 128.76450466723304],
                 [105.16730680744676, 108.90901096434611, 113.0917526727636, 116.38474219591838, 119.43574327244848,
                 122.85085041135066, 126.6630484766052, 128.66331171923127, 131.8785066161623]]

# 绘制折线图
plt.plot(range(len(socialWelfare[0])),  socialWelfare[0], label='V=0', marker='*', color='#d95319')
plt.plot(range(len(socialWelfare[1])),  socialWelfare[1], label='V=5', marker='o', color='green')
plt.plot(range(len(socialWelfare[2])),  socialWelfare[2], label='V=10(Our scheme)', marker='^')

plt.xlim(0, 8)

# 添加标题和轴标签
plt.xlabel('ECSP Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Social Welfare', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(socialWelfare[0])), range(1, 10), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 保存图像时设置dpi参数
plt.savefig("Fig_8b.png", dpi=300)
# 显示图形
plt.show()
