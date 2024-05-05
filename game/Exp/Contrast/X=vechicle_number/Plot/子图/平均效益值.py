import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
import numpy as np

U_user_v, U_S_t_v, U_vop_v = [2.929951276545613, 2.929951276545613, 2.929951276545613, 2.9720430330062255,
                              2.9720430330062255, 2.9720430330062255, 2.9720430330062255, 3.1266285981296327,
                              3.1266285981296327, 3.1266285981296327, 3.1266285981296327, 3.1266285981296327,
                              3.1266285981296327], [16.221873263888895, 16.279658841556554, 16.33696730581911,
                                                    15.811259676825664, 15.876537376312772, 15.94479099708856,
                                                    15.988457519721393, 14.546639611908812, 14.626544413173328,
                                                    14.670629800378023, 14.728665061125332, 14.765426936992435,
                                                    14.89828143061464], [-7.38855274664109, -1.4413260998969244,
                                                                         0.7277221967345078, 1.8713940231281638,
                                                                         3.0284362826979394, 4.154984741980421,
                                                                         4.415784184824833, 5.129263839674659,
                                                                         6.094979940144398, 6.3774994682900825,
                                                                         6.066094557230833, 5.888465754416272,
                                                                         7.325860622702053]

U_ECSP = np.array([[16.473749565972227, 17.07342554012346, 15.370320987654328],
                   [16.473749565972227, 17.188996695458776, 15.370320987654328],
                   [16.48418688384084, 17.303613623983892, 15.370320987654328],
                   [16.14209355490321, 16.82429418388407, 14.798225169767258],
                   [16.166349664014277, 16.954849582858287, 14.798225169767258],
                   [16.211811691519145, 17.091356824409864, 14.798225169767258],
                   [16.24875108284069, 17.17684175661076, 14.800073282832026],
                   [14.579757704131932, 15.385478866301408, 13.707800357516218],
                   [14.670963362375002, 15.508625391463664, 13.744463434882993],
                   [14.722101589351928, 15.572841197992084, 13.768418402763965],
                   [14.790094100961976, 15.654311293663495, 13.803018828587168],
                   [14.833494117938347, 15.704410223887189, 13.826443650097682],
                   [14.992011816716657, 15.87783774944592, 13.91872511178336]])

# -----------------------------------主图
# 自定义刻度标签格式函数
def format_func(value, tick_number):
    return f"{value:.0f}"
x_major_locator = MultipleLocator(3)
# 创建一个大的图形和主图
fig, ax = plt.subplots()
# 绘制主图中的三条曲线
ax.plot(range(len(U_user_v)), U_user_v, label='$U_{i}$', marker='*', color='#d95319')
ax.plot(range(len(U_S_t_v)), U_S_t_v, label='$U^{ecsp}_{j}$', marker='o')
ax.plot(range(len(U_vop_v)), U_vop_v, label='$U_{vop}$', marker='x', color='green')

ax.set_xlim(0, 9)
ax.set_ylim(-18, 17)
# 添加标题和轴标签
ax.set_xlabel('Vechicle Number', fontweight='bold', fontsize=15.5)
ax.set_ylabel('Average Utility', fontweight='bold', fontsize=15.5)
ax.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
# 添加主图的标签和图例
ax.legend()

# -----------------------------------子图 创建子图并调整位置
left, bottom, width, height = 0.5, 0.2, 0.35, 0.35
ax_inset = fig.add_axes([left, bottom, width, height])  # left, bottom, width, height

# 绘制子图中的三条曲线
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 0], label='$U_{j}(j=1)$', marker='.', color='#990066')
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 1], label='$U_{j}(j=2)$', marker='o', color='green')
ax_inset.plot(range(len(U_ECSP[:, 0])), U_ECSP[:, 2], label='$U_{j}(j=3)$', marker='*', color='y')


ax_inset.set_xlim(0, 9)
ax_inset.set_ylim(13, 19)
# 添加标题和轴标签
ax_inset.set_xlabel('User Number', fontweight='bold', fontsize=8)
ax_inset.set_ylabel('ECSP Utility', fontweight='bold', fontsize=8)
ax_inset.set_xticks(range(len(U_user_v)), range(0, 13), fontsize=8)  # 修改x轴刻度字体大小
# 设置y轴刻度字体大小
ax_inset.tick_params(axis='y', labelsize=8)
ax_inset.yaxis.set_major_formatter(FuncFormatter(format_func))
# 设置x轴刻度间隔为3
ax_inset.xaxis.set_major_locator(x_major_locator)
# 设置y轴刻度间隔为2
ax_inset.yaxis.set_major_locator( MultipleLocator(2))
ax_inset.legend(loc='upper right', fontsize=8)
# 显示图形
plt.show()
# 保存图像时设置dpi参数
# fig.savefig("output.png", dpi=300)
