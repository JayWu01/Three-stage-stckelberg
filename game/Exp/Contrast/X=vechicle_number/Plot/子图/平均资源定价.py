import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
# 预设字体格式，并传给rc方法
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
U_user_v, U_S_t_v = [2.0250000000000004, 2.0250000000000004, 2.0250000000000004, 1.9750000000000005, 1.9750000000000005,
                     1.9750000000000005, 1.9750000000000005, 1.8750000000000004, 1.8750000000000004, 1.8750000000000004,
                     1.8750000000000004, 1.8750000000000004, 1.8750000000000004], [1.2583867289596864,
                                                                                   1.143996962960242,
                                                                                   1.0912116804364465,
                                                                                   1.0905580311526903,
                                                                                   1.0562605302704282,
                                                                                   1.0233100297548898,
                                                                                   1.0039018842320633,
                                                                                   1.038632508610798, 1.011047133797538,
                                                                                   0.9972989689919762,
                                                                                   0.9804106372073879,
                                                                                   0.970308011973287,
                                                                                   0.9368084709346176]

P_ECSP = np.array([[1.6000000000000003, 2.400000000000001, 1.6500000000000004],
                   [1.6000000000000003, 2.400000000000001, 1.6500000000000004],
                   [1.6000000000000003, 2.400000000000001, 1.6500000000000004],
                   [1.6000000000000003, 2.3000000000000007, 1.6500000000000004],
                   [1.6000000000000003, 2.3000000000000007, 1.6500000000000004],
                   [1.6000000000000003, 2.3000000000000007, 1.6500000000000004],
                   [1.6000000000000003, 2.3000000000000007, 1.6500000000000004],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003]])

P_VOP = np.array([[1.1161263854644843, 1.4668085673739537, 1.1922252340406212],
                  [1.0945304133047047, 1.3060219044546768, 1.0314385711213445],
                  [1.061234828584595, 1.2434917730290387, 0.9689084396957061],
                  [1.005884636578934, 1.3511790206369039, 0.9146104362422331],
                  [0.9715871356966719, 1.316881519754642, 0.8803129353599712],
                  [0.9386366351811335, 1.2839310192391036, 0.8473624348444329],
                  [0.919228489658307, 1.2645228737162768, 0.8279542893216062],
                  [0.9778848528416368, 1.2589892299460135, 0.8790234430447436],
                  [0.9502994780283769, 1.2314038551327537, 0.8514380682314837],
                  [0.9365513132228152, 1.2176556903271918, 0.8376899034259218],
                  [0.9196629814382268, 1.2007673585426035, 0.8208015716413335],
                  [0.9095603562041257, 1.1906647333085025, 0.8106989464072325],
                  [0.8760608151654563, 1.1571651922698332, 0.7771994053685631]])

U_user_v = [np.mean(arr) for arr in U_user_v]  # 因为输出的是所有ECSP的定价，所以需要求平均
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.1f}"

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')


# plt.plot(range(len(P_ECSP[:, 0])), P_ECSP[:, 0], label='$p_{1}$', marker='.')
# plt.plot(range(len(P_ECSP[:, 1])), P_ECSP[:, 1], label='$p_{2}$', marker='o')
# plt.plot(range(len(P_ECSP[:, 2])), P_ECSP[:, 2], label='$p_{3}$', marker='s')
#
# plt.plot(range(len(P_VOP[:, 0])), P_VOP[:, 0], label='$p^{1}_{vop}$', marker='^')
# plt.plot(range(len(P_VOP[:, 1])), P_VOP[:, 1], label='$p^{2}_{vop}$', marker='x')
# plt.plot(range(len(P_VOP[:, 2])), P_VOP[:, 2], label='$p^{3}_{vop}$', marker='*', color='#d95319')

# 创建一个大的图形和主图
fig, ax = plt.subplots()
x_major_locator = MultipleLocator(3)
# 绘制主图中的曲线 绘制折线图
ax.plot(range(len(U_user_v)), U_user_v, label='$p_{j}$', marker='*', color='#d95319')
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$p^{j}_{vop}$', marker='o')
ax.set_xlim(0, 9)
# ax.set_ylim(0, 8)
# 添加标题和轴标签
ax.set_xlabel('User Number', fontweight='bold', fontsize=15.5)
ax.set_ylabel('Average Utility', fontweight='bold', fontsize=15.5)
ax.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
# 添加主图的标签和图例
ax.legend(loc='upper right', ncol=2)

# -----------------------------------子图1 创建子图并调整位置
left, bottom, width, height = 0.205, 0.35, 0.3, 0.3
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(P_ECSP[:, 0])), P_ECSP[:, 0], label='$p_{1}$', marker='.')
ax_inset.plot(range(len(P_ECSP[:, 1])), P_ECSP[:, 1], label='$p_{2}$', marker='o')
ax_inset.plot(range(len(P_ECSP[:, 2])), P_ECSP[:, 2], label='$p_{3}$', marker='s', color='#d95319')

ax_inset.set_xlim(0, 9)
# 添加标题和轴标签
ax_inset.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP Price', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
ax_inset.set_ylim(1.4, 2.8)
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
ax_inset.legend(loc='upper center', ncol=3, fontsize=8)
# -----------------------------------子图2 创建子图并调整位置
left, bottom, width, height = 0.59, 0.35, 0.3, 0.3
ax_inset2 = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset2.plot(range(len(P_VOP[:, 0])), P_VOP[:, 0], label='$p^{1}_{vop}$', marker='^')
ax_inset2.plot(range(len(P_VOP[:, 1])), P_VOP[:, 1], label='$p^{2}_{vop}$', marker='x')
ax_inset2.plot(range(len(P_VOP[:, 2])), P_VOP[:, 2], label='$p^{3}_{vop}$', marker='*', color='#d95319')

ax_inset2.set_xlim(0, 9)
# 设置y轴刻度标签格式
ax_inset2.yaxis.set_major_formatter(FuncFormatter(format_func))
# 添加标题和轴标签
ax_inset2.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset2.set_ylabel('VOP Price', fontweight='bold', fontsize=8)
ax_inset2.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset2.tick_params(axis='y', labelsize=8)
# 设置x轴刻度间隔为3
ax_inset2.xaxis.set_major_locator(x_major_locator)
ax_inset2.set_ylim(0.6, 1.8)
ax_inset2.legend(loc='upper center', ncol=3, fontsize=7)

# 显示图形
plt.show()
# 保存图像时设置dpi参数
# fig.savefig("output.png", dpi=300)
