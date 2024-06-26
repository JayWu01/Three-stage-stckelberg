import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
# 预设字体格式，并传给rc方法
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
U_user_v, U_S_t_v = [1.0833333333333333, 1.4166666666666667, 1.6500000000000004, 1.7500000000000007, 1.8833333333333337,
                     1.9833333333333336, 2.2166666666666672, 2.3166666666666678, 2.450000000000001,
                     2.450000000000001], [0.3784406255828519, 0.5948397197368225, 0.7936090283731675,
                                          0.9804106372073879, 1.1131814951706172, 1.2787265789707642,
                                          1.3247141051791673, 1.4757821370189108, 1.601964240172759, 1.7506947379203395]

P_ECSP = np.array([[0.8999999999999999, 1.4000000000000001, 0.9499999999999998], [1.2, 1.8000000000000005, 1.25],
                   [1.4000000000000001, 2.1000000000000005, 1.4500000000000002],
                   [1.5000000000000002, 2.2000000000000006, 1.5500000000000003],
                   [1.6000000000000003, 2.400000000000001, 1.6500000000000004],
                   [1.7000000000000004, 2.500000000000001, 1.7500000000000004],
                   [1.9000000000000006, 2.800000000000001, 1.9500000000000006],
                   [2.0000000000000004, 2.9000000000000012, 2.0500000000000007],
                   [2.1000000000000005, 3.1000000000000014, 2.150000000000001],
                   [2.1000000000000005, 3.1000000000000014, 2.150000000000001]])

P_VOP = np.array([[0.380655576540493, 0.41955328377153345, 0.33511301643652947],
                  [0.5764421888726253, 0.6959792259096615, 0.5120977444281807],
                  [0.7202982730365506, 1.0199411301794068, 0.6405876819035454],
                  [0.9196629814382268, 1.2007673585426035, 0.8208015716413335],
                  [1.0771387763153983, 1.3002464152042865, 0.9621592939921663],
                  [1.1997234043675897, 1.5649234043675895, 1.0715329281771133],
                  [1.2430475545433186, 1.6197967191214295, 1.1112980418727538],
                  [1.3434105206590692, 1.8823492179770767, 1.2015866724205866],
                  [1.5032845702552806, 1.951591021868182, 1.3510171283948145],
                  [1.6428898634212925, 2.1351035517640113, 1.4740907985757148]])

U_user_v = [np.mean(arr) for arr in U_user_v]  # 因为输出的是所有ECSP的定价，所以需要求平均
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
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.0f}"
x_major_locator = MultipleLocator(3)
# 创建一个大的图形和主图
fig, ax = plt.subplots()
# 绘制主图中的曲线 绘制折线图
ax.plot(range(len(U_user_v)), U_user_v, label='$p_{j}$', marker='*', color='#d95319',linewidth=4,markersize=10)
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$p^{j}_{vop}$', marker='o',linewidth=4,markersize=10)
ax.set_xlim(0, 9)
ax.set_ylim(0, 5)
# 添加标题和轴标签
ax.set_xlabel('User Number', fontweight='bold', fontsize=20)
ax.set_ylabel('Average Price', fontweight='bold', fontsize=20)
ax.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=20)  # 修改x轴刻度字体大小
ax.tick_params(axis='y', labelsize=20)  # 修改y轴刻度字体大小
ax.figure.subplots_adjust(bottom=0.14, left=0.12, right=0.965, top=0.97)
# 添加主图的标签和图例
ax.legend(loc='lower right',ncol=2,fontsize=17.5)

# -----------------------------------子图1 创建子图并调整位置
left, bottom, width, height = 0.2, 0.62, 0.3, 0.3
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(P_ECSP[:, 0])), P_ECSP[:, 0], label='$p_{1}$', marker='.', color='#990066',linewidth=2,markersize=10)
ax_inset.plot(range(len(P_ECSP[:, 1])), P_ECSP[:, 1], label='$p_{2}$', marker='*', color='green',linewidth=2,markersize=10)
ax_inset.plot(range(len(P_ECSP[:, 2])), P_ECSP[:, 2], label='$p_{3}$', marker='x', color='y',linewidth=2,markersize=10)

ax_inset.set_xlim(0, 9)
# 添加标题和轴标签
ax_inset.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP Price', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
ax_inset.set_ylim(0.7, 4)
ax_inset.legend(loc='upper center', ncol=3, fontsize=8)
# -----------------------------------子图2 创建子图并调整位置
left, bottom, width, height = 0.58, 0.62, 0.3, 0.3
ax_inset2 = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset2.plot(range(len(P_VOP[:, 0])), P_VOP[:, 0], label='$p^{1}_{vop}$', marker='.', color='#990066',linewidth=2,markersize=10)
ax_inset2.plot(range(len(P_VOP[:, 1])), P_VOP[:, 1], label='$p^{2}_{vop}$', marker='*', color='green',linewidth=2,markersize=10)
ax_inset2.plot(range(len(P_VOP[:, 2])), P_VOP[:, 2], label='$p^{3}_{vop}$', marker='x', color='y',linewidth=2,markersize=10)

ax_inset2.set_xlim(0, 9)
# 添加标题和轴标签
ax_inset2.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset2.set_ylabel('VOP Price', fontweight='bold', fontsize=8)
ax_inset2.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset2.tick_params(axis='y', labelsize=8)
ax_inset2.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset2.xaxis.set_major_locator(x_major_locator)
ax_inset2.set_ylim(-0.1, 3)
ax_inset2.legend(loc='upper center', ncol=3, fontsize=7)

# 显示图形
plt.show()
# 保存图像时设置dpi参数
fig.savefig("Fig_4b.png", dpi=300)
