import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np
from numpy import array

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)

U_user_v, U_S_t_v =[1.0833333333333333, 1.4166666666666667, 1.6500000000000004, 1.7500000000000007, 1.8833333333333337, 1.9833333333333336, 2.2166666666666672, 2.3166666666666678, 2.450000000000001, 2.450000000000001],[0.3784406255828519, 0.5948397197368225, 0.7936090283731675, 0.9804106372073879, 1.1131814951706172, 1.2787265789707642, 1.3247141051791673, 1.4757821370189108, 1.601964240172759, 1.7506947379203395]

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
plt.xlabel('user数量', fontsize=16)
plt.ylabel('资源定价', fontsize=16)
plt.xticks(fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
