import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
import numpy as np
# 四组数据
np.set_printoptions(precision=16)

U_user_v, U_S_t_v = [4.597423187598625, 3.5678074074074067, 3.0976765188834134, 2.9069000760291077, 2.671810606060606,
                     2.528101587301587, 2.245374348569836, 2.1645899448649644, 2.113898474618654, 2.078565800180203], [
                        1.4920704135791139, 2.315827716320445, 3.0160057942184064, 3.773678957854967,
                        4.3356131977736645, 4.922895641492554, 5.101074360136173, 5.620053917744894, 6.174495722051909,
                        6.745880417143311]

Resouce_ECSP = np.array([[1.9992341482024836, 1.3885941464116458, 1.088382946123212],
                [2.815600666991946, 2.6031707038663354, 1.5287117781030535],
                [3.0583202535547045, 4.5255886982059135, 1.4641084308946013],
                [4.1948885193836, 4.908488030735567, 2.2176603234457346],
                [5.230141140358704, 4.846146959068232, 2.9305514938940576],
                [5.472198579314881, 6.3880992896574345, 2.9083890555053458],
                [5.668288675215482, 6.601635983388853, 3.033298421804183],
                [5.722900697929706, 8.250837322144925, 2.8864237331600506],
                [6.8343085948944164, 7.900218813576224, 3.788959757685088],
                [7.476594265965723, 8.66043401641004, 4.100612969054168]])

# -----------------------------------主图
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.0f}"
x_major_locator = MultipleLocator(3)
# 创建一个大的图形和主图
fig, ax = plt.subplots()
# 绘制折线图
ax.plot(range(len(U_user_v)), U_user_v, label='$f_i$', marker='*', color='#d95319')
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$f^{j}_{vop}$', marker='o')
ax.set_xlim(0, 9)
ax.set_ylim(0, 12)
# 添加标题和轴标签
ax.set_xlabel('User Number', fontweight='bold', fontsize=15.5)
ax.set_ylabel('Average Resource Purchase', fontweight='bold', fontsize=15.5)
ax.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=13.5)  # 修改x轴刻度字体大小
# 添加主图的标签和图例
ax.legend()

# -----------------------------------子图 创建子图并调整位置
left, bottom, width, height = 0.21, 0.48, 0.35, 0.35
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 0], label='$f^{1}_{vop}$', marker='.', color='#990066')
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 1], label='$f^{2}_{vop}$', marker='+', color='green')
ax_inset.plot(range(len(Resouce_ECSP[:, 0])), Resouce_ECSP[:, 2], label='$f^{3}_{vop}$', marker='x', color='y')

ax_inset.set_xlim(0, 9)
ax_inset.set_ylim(0, 12)
# 添加标题和轴标签
ax_inset.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP\'s Resource Purchase to VOP', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
# 设置y轴刻度间隔为2
ax_inset.yaxis.set_major_locator( MultipleLocator(2))
ax_inset.legend(loc='upper center', ncol=3,fontsize=8)
# 显示图形
plt.show()
# 保存图像时设置dpi参数
fig.savefig("Fig_4c.png", dpi=300)