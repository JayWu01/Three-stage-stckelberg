import matplotlib.pyplot as plt
# 预设字体格式，并传给rc方法
import numpy as np
from matplotlib.ticker import MultipleLocator

# 四组数据
np.set_printoptions(precision=16)
U_user_v, U_S_t_v, U_vop_v = [9.098345528870142, 7.868087828956858, 6.946495359721777, 6.228250148191359,
                              5.651727446387, 5.178217857419935, 4.782078899434659, 4.445603405192788,
                              4.156147185607261, 3.904428032957579, 3.7934886146592306, 3.5807617428836593,
                              3.4927334323649633, 3.4175807482494585, 3.3537841283415255, 3.1782054806455085,
                              3.1266285981296327], [[-110.87629191947813, -125.93730912555179, -62.29466206073565],
                                                    [-62.50949432173525, -100.11861973232021, -34.658267441468894],
                                                    [-35.25343659897895, -81.02175965662278, -18.312295360245926],
                                                    [-18.6012489215474, -66.20703639347691, -8.049800729028945],
                                                    [-7.871118566904947, -54.266632518839614, -1.3537200953990416],
                                                    [-0.7094059932366772, -44.339067729223, 3.114958077414563],
                                                    [4.174875189545183, -35.870777598462915, 6.122103720551598],
                                                    [7.544685009727299, -28.59098543480587, 8.068838795061728],
                                                    [9.864665668241628, -22.129939843849534, 9.326565462581918],
                                                    [10.036491268340736, -12.612902780823902, 9.685995766228046],
                                                    [11.690159154566125, -4.892662650475955, 10.71422093040325],
                                                    [12.166691039766459, 0.6562934792165045, 11.226016118377594],
                                                    [12.434304065299234, 6.650018425924465, 11.64839994595041],
                                                    [12.578270528282236, 10.513733859738428, 12.021667954843975],
                                                    [14.139123423624264, 13.970020291269668, 13.148703437016714],
                                                    [14.904104376392098, 14.616891590725574, 13.843843655056526],
                                                    [14.790094100247682, 15.654311292827693, 13.803018828209554]], [
                                 78.57544123026389, 57.46170032194149, 45.75525825648279, 38.538901351995534,
                                 33.749281840145265, 30.392412087604146, 27.939256339466695, 25.361407179125198,
                                 23.38910232339948, 21.86561916471495, 16.94448078047614, 15.987086837539,
                                 12.43558051490588, 9.508316880334538, 7.607609839424171, 7.296842172159765,
                                 6.066094540991868]

# 绘制折线图
# plt.plot(range(1, len(V2) + 1), V2, label=r'$\theta_{2}$', c='#0066cc', marker='o')
# plt.axvline(x=2, ymin=0, ymax=0.55, color='#0066cc', linestyle='--')
#
# plt.plot(range(1, len(V4) + 1), V4, label=r'$\theta_{4}$', c='#990066', marker='*')
# plt.axvline(x=4, ymin=0, ymax=0.57, color='#990066', linestyle='--')
#
# plt.plot(range(1, len(V6) + 1), V6, label=r'$\theta_{6}$', c='#d95319', marker='<')
# plt.axvline(x=6, ymin=0, ymax=0.60, color='#d95319', linestyle='--')
#
# plt.plot(range(1, len(V8) + 1), V8, label=r'$\theta_8$', c='y', marker='.')
# plt.axvline(x=8, ymin=0, ymax=0.65, color='y', linestyle='--')
#
# plt.plot(range(1, len(V10) + 1), V10, label=r'$\theta_{10}$', c='#336600', marker='s')
# plt.axvline(x=10, ymin=0, ymax=0.77, color='#336600', linestyle='--')


U_S_t_v = np.array(U_S_t_v)
# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label=r'$\bar{U}_{i}$', c='#0066cc', marker='o', linewidth=3, markersize=10)
plt.plot(range(len(U_S_t_v[:, 0])), U_S_t_v[:, 0], label='$U_{j}(j=1)$', c='#990066', marker='*', linewidth=3,
         markersize=10)
plt.plot(range(len(U_S_t_v[:, 1])), U_S_t_v[:, 1], label='$U_{j}(j=2)$', c='#d95319', marker='<', linewidth=3,
         markersize=10)
plt.plot(range(len(U_S_t_v[:, 2])), U_S_t_v[:, 2], label='$U_{j}(j=3)$', c='y', marker='.', linewidth=3, markersize=10)
plt.plot(range(len(U_vop_v)), U_vop_v, label='$U_{vop}$', c='#336600', marker='s', linewidth=3, markersize=10)
plt.subplots_adjust(left=0.135)

# 添加标题和轴标签
plt.xlabel('The Number Of Game Rounds', fontweight='bold', fontsize=20)
plt.ylabel('Utility', fontweight='bold', fontsize=20)
plt.xticks(range(len(U_user_v)), fontsize=20)  # 修改x轴刻度字体大小
plt.yticks(fontsize=20)  # 修改y轴刻度字体大小
plt.gca().xaxis.set_major_locator(MultipleLocator(2)) #让x轴的显示刻度的间隔为3
plt.subplots_adjust(bottom=0.14, left=0.185, right=0.965, top=0.97)
# 添加图例并设置字体大小
plt.legend(fontsize='16.5')
# 显示图形

# 保存图像时设置dpi参数
plt.savefig("Fig_3.png", dpi=300)
plt.show()
