import matplotlib.pyplot as plt
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
plt.figure(1)
socialWelfare = [[104.16466780433962, 101.63910423705691, 100.12796887802119, 98.88994072144799, 104.66186426440885,
                  105.34802425243409, 110.86269554887856, 106.78718981449744, 95.64683665570371],
                 [104.97567099862871, 106.93431552683454, 111.21938868946728, 114.64539331691222, 117.43557578889813,
                  119.85208663898406, 124.78170655479589, 125.83472284579223, 128.76450466723304],
                 [105.16730680744676, 108.90901096434611, 113.0917526727636, 116.38474219591838, 119.43574327244848,
                  122.85085041135066, 126.6630484766052, 128.66331171923127, 131.8785066161623]]

# 绘制折线图
# plt.plot(range(1, 10), socialWelfare[0], label='Without Vechicle', marker='.')
# plt.plot(range(1, 10), socialWelfare[1], label='The number of Vechicle is =5', marker='o')
# plt.plot(range(1, 10), socialWelfare[2], label='The number of Vechicle is =10', marker='^')

# 设置条形图的宽度和间隔
bar_width = 0.15
index = np.arange(9)
# 创建条形图 #332c83 0899c7 a4bb6a efef6d
plt.bar(index, socialWelfare[0], bar_width, label='V=0', color='#4dbeee', zorder=2, edgecolor='black',
        linewidth=0.5)
plt.bar(index + bar_width, socialWelfare[1], bar_width, label='V=5', color='#77ac30', zorder=2,
        edgecolor='black',
        linewidth=0.5)
plt.bar(index + 2 * bar_width, socialWelfare[2], bar_width, label='V=10(Our scheme)', color='#d95319',
        zorder=2, edgecolor='black',
        linewidth=0.5)
# plt.bar(index + 3 * bar_width, U_D_edge_cloud_idle, bar_width, label='IVVEC',color='#0072bd',zorder=2, edgecolor='black',
#         linewidth=0.5)

# 添加标题和轴标签
plt.xlabel('ECSP Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Social Welfare', fontweight='bold', fontsize=15.5)
# 旋转刻度标签并间隔显示
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # 每隔2个类别显示一个刻度标签
plt.xticks(index + 1.5 * bar_width, range(1, 10))
# 设置 y 轴的上下限
plt.ylim(0, 150)  # 这里将 y 轴范围设置为 0 到 3.5
# 旋转 x 轴刻度标签为垂直显示
plt.xticks(fontsize=13.5)
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4)
# 保存图像时设置dpi参数
plt.savefig("Fig_8b.png", dpi=300)
# 显示图形
plt.show()
