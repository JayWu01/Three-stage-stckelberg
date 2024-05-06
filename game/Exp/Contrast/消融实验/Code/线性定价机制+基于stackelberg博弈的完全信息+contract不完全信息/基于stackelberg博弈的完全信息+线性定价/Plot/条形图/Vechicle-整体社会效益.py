import matplotlib.pyplot as plt

# 预设字体格式，并传给rc方法
import numpy as np
# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [[100.12796887802119, 106.19076668010068, 108.53256967558933, 109.1990348436584, 110.57933143826808,
                  111.92889329377228, 112.44745879798933, 111.80379317505611, 112.91576987801949, 113.48704372270964,
                  113.49689457861072, 113.68633508671272, 115.29664418564332],
                 [100.12796887802119, 106.19074056307332, 108.62293932900043, 109.49602946390058, 111.13651417199685,
                  112.46930597436842, 111.92044186675678, 112.83575817319243, 113.71838522341044, 114.5827439153567,
                  115.47927833727186, 116.38177135448406, 96.71941265932986],
                 [100.12796887802119, 103.72668668152207, 106.28626133938467, 109.32346354014297, 109.33956978766159,
                  100.08038357277965, 100.016890534361, 110.5087294225272, 100.1743315044348, 107.49552827799562,
                  106.04595464692619, 111.03448936778115, 105.2394387630879]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')
# plt.plot(range(1, 10), socialWelfare[0], label='基于stackelberg的完全信息', marker='.')
# plt.plot(range(1, 10), socialWelfare[1], label='线性定价-p=1.2', marker='o')
# plt.plot(range(1, 10), socialWelfare[2], label='线性定价-p=1.5', marker='*')
# plt.plot(range(1, 10), socialWelfare[3], label='基于合同的不完全信息', marker='^')

# 绘制折线图
# plt.plot(range(0, 13), socialWelfare[0], label='IA-C(Our scheme)', marker='^')
# plt.plot(range(0, 13), socialWelfare[1], label='NIA-S', marker='.')
# plt.plot(range(0, 13), socialWelfare[2], label='LP', marker='o')
# # 添加图例
# plt.legend()

# 设置条形图的宽度和间隔
bar_width = 0.2
index = np.arange(0, 13)
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
plt.xlabel('Vechicle Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Social Welfare',fontweight='bold', fontsize=15.5)
# 旋转刻度标签并间隔显示
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # 每隔2个类别显示一个刻度标签
plt.xticks(index + 1.5 * bar_width, range(0, 13))
# 设置 y 轴的上下限
plt.ylim(0, 135)  # 这里将 y 轴范围设置为 0 到 3.5
# 旋转 x 轴刻度标签为垂直显示
plt.xticks(fontsize=13.5)
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
# 保存图像时设置dpi参数
plt.savefig("Fig_7c.png", dpi=300)
# 显示图形
plt.show()