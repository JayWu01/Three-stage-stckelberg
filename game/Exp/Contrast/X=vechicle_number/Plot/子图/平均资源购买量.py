import matplotlib.pyplot as plt
# 预设字体格式，并传给rc方法
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

# 四组数据
np.set_printoptions(precision=16)

U_user_v, U_S_t_v = [2.7314583333333333, 2.7314583333333333, 2.7314583333333333, 2.722989038940125, 2.722989038940125,
                     2.722989038940125, 2.722989038940125, 2.9069000760291077, 2.9069000760291077, 2.9069000760291077,
                     2.9069000760291077, 2.9069000760291077, 2.9069000760291077], [0.0, 0.7601682555043867,
                                                                                   1.0728189126325773,
                                                                                   1.817534848506293, 1.989022352917603,
                                                                                   2.1537748554952945,
                                                                                   2.3867609053815784,
                                                                                   2.6897461060394994,
                                                                                   3.103526728238397, 3.309749200321826,
                                                                                   3.563074177090651, 3.714613555602166,
                                                                                   4.217106671182205]

Resouce_ECSP = np.array([[0.0, 0.0, 0.0], [0.0, 1.5203365110087734, 0.0], [0.64613676164144, 2.1456378252651547, 0.0],
                         [0.36425171286577296, 3.635069697012586, 0.0], [1.0502017305110165, 3.978044705835206, 0.0],
                         [1.7092117408217824, 4.307549710990589, 0.0],
                         [2.097374651278315, 4.501631166218857, 0.2718906445442997],
                         [3.030451091315399, 4.3262693167014685, 1.0532228953775302],
                         [3.582158587580601, 4.602123064834066, 1.6049303916427284],
                         [3.8571218836918355, 4.739604712889685, 1.8798936877539667],
                         [4.1948885193836, 4.908488030735567, 2.2176603234457346],
                         [4.396941024065622, 5.009514283076578, 2.4197128281277536],
                         [5.0669318448390115, 5.344509693463271, 3.089703648901139]])


# -----------------------------------主图
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.1f}"


# 创建一个大的图形和主图
fig, ax = plt.subplots()
x_major_locator = MultipleLocator(3)
# 绘制折线图
ax.plot(range(len(U_user_v)), U_user_v, label='$f_i$', marker='*', color='#d95319',linewidth=4,markersize=10)
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$f^{j}_{vop}$', marker='o',linewidth=4,markersize=10)
ax.set_xlim(0, 12)
ax.set_ylim(-2, 4.5)
# 添加标题和轴标签
ax.set_xlabel('Vechicle Number', fontweight='bold', fontsize=20)
ax.set_ylabel('Average Resource Purchase', fontweight='bold', fontsize=20)
ax.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=20)  # 修改x轴刻度字体大小
# 添加主图的标签和图例
ax.legend(loc='upper left', ncol=2,fontsize=17.5)
ax.tick_params(axis='y', labelsize=20)  # 修改y轴刻度字体大小
# 添加主图的标签和图例
ax.figure.subplots_adjust(bottom=0.14, left=0.16, right=0.965, top=0.97)

# -----------------------------------子图 创建子图并调整位置
left, bottom, width, height = 0.51, 0.22, 0.35, 0.35
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 0], label='$f^{1}_{vop}$', marker='.', color='#990066',linewidth=2,markersize=10)
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 1], label='$f^{2}_{vop}$', marker='*', color='green',linewidth=2,markersize=10)
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 2], label='$f^{3}_{vop}$', marker='x', color='y',linewidth=2,markersize=10)

ax_inset.set_xlim(0, 9)
ax_inset.set_ylim(-0.2, 8)
# 设置y轴刻度标签格式
# 添加标题和轴标签
ax_inset.set_xlabel('Vechicle Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP\'s Resource Purchase to VOP', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=8)  # 修改x轴刻度字体大小
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
ax_inset.legend(loc='upper center', ncol=3, fontsize=8)
# 显示图形
plt.show()
# 保存图像时设置dpi参数
fig.savefig("Fig_6c.png", dpi=300)
