import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
U_user_v, U_S_t_v, U_vop_v = [4.446175506742655, 3.695959251718796, 3.3560239647087537, 3.1266285981296327,
                              2.810665575000767, 2.7129759519862087, 2.4179068202155336, 2.398926291315448,
                              2.3720723742150187, 2.311940957465355], [3.4356828992203714, 7.783528623816274,
                                                                       11.848873440756568, 14.749141407737547,
                                                                       17.82519758748793, 20.53470142991458,
                                                                       26.556771266340263, 29.75162665016549,
                                                                       34.15105599500961, 34.249211231658364], [
                                 0.8475742403409048, 2.144062817939837, 4.239640238488973, 6.066094557230833,
                                 7.486556260650842, 10.30827288531596, 11.049425908999307, 14.475436553734891,
                                 16.07045486722827, 19.22405029864413]

U_ECSP = np.array([[3.5574011005701958, 3.563000262143963, 3.1866473349469566],
                   [7.9641130174053085, 8.198163705975182, 7.188309148068331],
                   [12.236110493722173, 12.153446487090557, 11.157063341456972],
                   [14.790094100961976, 15.654311293663495, 13.803018828587168],
                   [17.56892440436155, 19.157972198771482, 16.74869615933077],
                   [20.294846154508626, 21.832854271170557, 19.476403864064565],
                   [26.349159020997543, 28.627071938477297, 24.69408283954595],
                   [29.669165612428245, 31.731240731507555, 27.854473606560674],
                   [33.54744434925621, 36.82040179331902, 32.085321842453574],
                   [33.32781279611168, 36.74556147649903, 32.674259422364386]])

# -----------------------------------主图
# 创建一个大的图形和主图
fig, ax = plt.subplots()
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.0f}"
x_major_locator = MultipleLocator(3)
# 绘制主图中的三条曲线
ax.plot(range(len(U_user_v)), U_user_v, label='$U_{i}$', marker='*', color='#d95319',linewidth=4,markersize=10)
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$U^{ecsp}_{j}$', marker='o',linewidth=4,markersize=10)
ax.plot(range(len(U_vop_v)), U_vop_v, label='$U_{vop}$', marker='^', color='green',linewidth=4,markersize=10)
ax.set_xlim(0, 9)
ax.set_ylim(0, 60)
# 添加标题和轴标签
ax.set_xlabel('User Number', fontweight='bold', fontsize=20)
ax.set_ylabel('Average Utility', fontweight='bold', fontsize=20)
ax.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=20)  # 修改x轴刻度字体大小
ax.tick_params(axis='y', labelsize=20)  # 修改y轴刻度字体大小
# 添加主图的标签和图例
ax.legend(loc='upper right',fontsize=17.5)
ax.figure.subplots_adjust(bottom=0.14, left=0.12, right=0.965, top=0.97)

# -----------------------------------子图 创建子图并调整位置
left, bottom, width, height = 0.22, 0.5, 0.35, 0.35
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 0], label='$U_{j}(j=1)$', marker='.', color='#990066',linewidth=2,markersize=10)
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 1], label='$U_{j}(j=2)$', marker='*', color='green',linewidth=2,markersize=10)
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 2], label='$U_{j}(j=3)$', marker='x', color='y',linewidth=2,markersize=10)

ax_inset.set_xlim(0, 9)
# 添加标题和轴标签
ax_inset.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP Utility', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=8)  # 修改x轴刻度字体大小
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
ax_inset.legend(fontsize=8)
# 显示图形
plt.show()
# 保存图像时设置dpi参数
fig.savefig("Fig_4a.png", dpi=300)
