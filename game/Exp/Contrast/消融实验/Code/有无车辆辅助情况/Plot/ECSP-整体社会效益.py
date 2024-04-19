import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)
# socialWelfare = [[100.12796887802119, 104.66186426440885, 110.86269554887856, 95.64683665570371, 107.3944723963042],
#                   [111.21938871788777, 117.43557585617309, 124.78180250097209, 128.76451339623873, 136.9238275377318],
#                   [113.09175271793909, 119.4357435404227, 126.66653324666862, 131.8787893926613, 145.570401879881]]

socialWelfare = [[104.16466780433962, 101.63910423705691, 100.12796887802119, 98.88994072144799, 104.66186426440885,
                 105.34802425243409, 110.86269554887856, 106.78718981449744, 95.64683665570371],
                 [104.97567099862871, 106.93431552683454, 111.21938868946728, 114.64539331691222, 117.43557578889813,
                 119.85208663898406, 124.78170655479589, 125.83472284579223, 128.76450466723304],
                 [105.16730680744676, 108.90901096434611, 113.0917526727636, 116.38474219591838, 119.43574327244848,
                 122.85085041135066, 126.6630484766052, 128.66331171923127, 131.8785066161623]]


# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(1, 10), socialWelfare[0], label='Without Vechicle', marker='.')
plt.plot(range(1, 10), socialWelfare[1], label='The number of Vechicle is =5', marker='o')
plt.plot(range(1, 10), socialWelfare[2], label='The number of Vechicle is =10', marker='^')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('ECSP数量', fontsize=16)
plt.ylabel('整体社会效益', fontsize=16)
plt.xticks(range(1, 11), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
