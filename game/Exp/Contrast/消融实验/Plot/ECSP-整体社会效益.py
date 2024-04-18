import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [[100.12796887802119, 104.66186426440885, 110.86269554887856, 95.64683665570371,107.3944723963042], [110.98826924235172,
                  117.22847985196667, 124.58400198319696, 128.57383411040254,136.76023540162637], [112.51713493310353, 118.89648474170168,
                  126.16467612924201, 131.38098959499257,145.37739970236356]]
# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(3, 12, 2),socialWelfare[0], label='Without Vechicle', marker='.')
plt.plot(range(3, 12, 2),socialWelfare[1], label='The number of Vechicle is =5', marker='o')
plt.plot(range(3, 12, 2),socialWelfare[2], label='The number of Vechicle is =10', marker='^')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('ECSP数量', fontsize=16)
plt.ylabel('整体社会效益', fontsize=16)
plt.xticks(range(3, 12, 2),fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
