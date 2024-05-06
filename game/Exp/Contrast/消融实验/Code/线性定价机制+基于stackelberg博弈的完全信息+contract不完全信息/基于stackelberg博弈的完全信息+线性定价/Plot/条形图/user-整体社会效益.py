import matplotlib.pyplot as plt
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [
    [33.487242176288525, 62.699335251304184, 90.5423243813811, 113.49689457861072, 132.08784397417617,
     154.4092008940024, 176.5356471855541, 201.13081615418938, 227.00918058348435, 239.64841625674418],
    [33.79715358638257, 62.5725283997306, 91.80954622237273, 115.47927833727186, 134.70453505219447, 157.78235579340617,
     177.35816889739533, 202.29905698318657, 232.31557675389396, 245.98314057279828],
    [30.70255582208295, 55.47885173470603, 79.227384122443, 106.04595464692619, 125.92503192158115, 137.84802615179146,
     163.05339025022676, 179.55012124747736, 197.16733729103044, 208.13678004040102]]

# # 绘制折线图
# plt.plot(range(len(socialWelfare[0])), socialWelfare[0], label='IA-C(Our scheme)', marker='*', color='#d95319')
# plt.plot(range(len(socialWelfare[1])), socialWelfare[1], label='NIA-S', marker='o')
# plt.plot(range(len(socialWelfare[2])), socialWelfare[2], label='LP', marker='^', color='green')
# # 添加图例
# plt.legend()
# plt.xlim(0, 9)
# # 添加标题和轴标签
# plt.xlabel('User Number', fontweight='bold', fontsize=15.5)
# plt.ylabel('SocialWelfare', fontweight='bold', fontsize=15.5)
# plt.xticks(range(len(socialWelfare[0])), range(5, 55, 5), fontsize=13.5)  # 修改x轴刻度字体大小
# plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# # 添加图例并设置字体大小
# plt.legend(fontsize='16')
# # 显示图形
#
# plt.show()

# 设置条形图的宽度和间隔
bar_width = 0.85
index = np.arange(5, 55, 5)
# 创建条形图 #332c83 0899c7 a4bb6a efef6d
plt.bar(index, socialWelfare[0], bar_width, label='IA-C(Our scheme)', color='#4dbeee', zorder=2, edgecolor='black',
        linewidth=0.5)
plt.bar(index + bar_width, socialWelfare[1], bar_width, label='NIA-S', color='#77ac30', zorder=2,
        edgecolor='black',
        linewidth=0.5)
plt.bar(index + 2 * bar_width, socialWelfare[2], bar_width, label='LP', color='#d95319',
        zorder=2, edgecolor='black',
        linewidth=0.5)
# 添加标题和轴标签
plt.xlabel('User Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Social Welfare', fontweight='bold', fontsize=15.5)
# 旋转刻度标签并间隔显示
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # 每隔2个类别显示一个刻度标签
plt.xticks(index + 1.5 * bar_width, range(5, 55, 5))
# 设置 y 轴的上下限
plt.ylim(0, 280)  # 这里将 y 轴范围设置为 0 到 3.5
# 旋转 x 轴刻度标签为垂直显示
plt.xticks( fontsize=13.5)
plt.yticks( fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4)
# 保存图像时设置dpi参数
plt.savefig("Fig_7a.png", dpi=300)
# 显示图形
plt.show()
