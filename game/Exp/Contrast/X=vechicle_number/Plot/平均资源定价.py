import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np
from numpy import array

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
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

U_user_v = [np.mean(arr) for arr in U_user_v]
# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label='ECSP的资源定价', marker='.')
plt.plot(range(len(U_S_t_v)), U_S_t_v, label='VOP的资源定价', marker='o')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('车辆数量', fontsize=16)
plt.ylabel('资源定价', fontsize=16)
plt.xticks(fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
