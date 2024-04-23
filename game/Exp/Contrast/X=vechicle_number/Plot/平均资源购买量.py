import matplotlib.pyplot as plt
import numpy as np
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

# U_user_v = [np.sum(arr) for arr in U_user_v]
# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label='$f_i$', marker='*', color='#d95319')
plt.plot(range(len(U_S_t_v)), U_S_t_v, label='$f^{j}_{vop}$', marker='o')

plt.xlim(0, 12)
# 添加标题和轴标签
plt.xlabel('Vechicle Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Average Resource Purchase', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(U_user_v)), range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
