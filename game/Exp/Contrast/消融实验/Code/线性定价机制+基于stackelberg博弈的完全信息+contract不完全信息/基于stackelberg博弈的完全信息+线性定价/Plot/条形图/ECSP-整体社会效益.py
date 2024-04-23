import matplotlib.pyplot as plt
# 预设字体格式，并传给rc方法
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
# socialWelfare = [[100.12796887802119, 104.66186426440885, 110.86269554887856, 95.64683665570371, 107.3944723963042],
#                   [111.21938871788777, 117.43557585617309, 124.78180250097209, 128.76451339623873, 136.9238275377318],
#                   [113.09175271793909, 119.4357435404227, 126.66653324666862, 131.8787893926613, 145.570401879881]]

socialWelfare = [[105.37979814022239, 109.28704913561442, 113.49689457861072, 115.47885272133635, 119.9463085295634,
                  126.6845855791226, 126.94241740651809, 138.03663445981388, 131.9631391389515, 132.82065009567182],
                 [105.52783167551948, 111.33543585141521, 115.47927833727186, 118.56929963249878, 116.94487359338252,
                  119.61213983227024, 115.84502097577459, 120.53394400972698, 135.31419256282513, 108.49751790451126],
                 [104.73622976152245, 100.36673629332606, 106.04595464692619, 113.40082430356924, 116.13708975599914,
                  117.00368069657723, 118.60429540740748, 123.91936404967883, 126.39027136852386, 126.27658051214108]]

# 设置条形图的宽度和间隔
bar_width = 0.15
index = np.arange(1, 11)
# 创建条形图 #332c83 0899c7 a4bb6a efef6d
plt.bar(index, socialWelfare[0], bar_width, label='IA-C', color='#4dbeee', zorder=2, edgecolor='black',
        linewidth=0.5)
plt.bar(index + bar_width, socialWelfare[1], bar_width, label='NIA-S', color='#77ac30', zorder=2,
        edgecolor='black',
        linewidth=0.5)
plt.bar(index + 2 * bar_width, socialWelfare[2], bar_width, label='LP', color='#d95319',
        zorder=2, edgecolor='black',
        linewidth=0.5)
# 添加标题和轴标签
plt.xlabel('ECSP Number', fontweight='bold', fontsize=15.5)
plt.ylabel('SocialWelfare', fontweight='bold', fontsize=15.5)
# 旋转刻度标签并间隔显示
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # 每隔2个类别显示一个刻度标签
plt.xticks(index + 1.5 * bar_width, range(1, 11))
# 设置 y 轴的上下限
plt.ylim(0, 160)  # 这里将 y 轴范围设置为 0 到 3.5
# 旋转 x 轴刻度标签为垂直显示
plt.xticks( fontsize=13.5)
plt.yticks( fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4)
# 显示图形

plt.show()
