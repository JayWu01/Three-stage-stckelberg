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

socialWelfare = [[105.37979814022239, 109.28704913561442, 113.49689457861072, 115.47885272133635, 119.9463085295634,
                  126.6845855791226, 126.94241740651809, 138.03663445981388, 131.9631391389515, 132.82065009567182],
                 [105.52783167551948, 111.33543585141521, 115.47927833727186, 118.56929963249878, 116.94487359338252,
                  119.61213983227024, 115.84502097577459, 120.53394400972698, 135.31419256282513, 108.49751790451126],
                 [104.73622976152245, 102.56158728588612, 103.29350763176198, 108.68585602066902, 104.91538033084652,
                  116.94841981498152, 122.42451916165233, 116.23140850494175, 99.36033014650889, 129.46188604691088]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(1, 11), socialWelfare[0], label='基于合同的不完全信息', marker='^')
plt.plot(range(1, 11), socialWelfare[1], label='基于stackelberg的完全信息', marker='.', color='red')
plt.plot(range(1, 11), socialWelfare[2], label='线性定价', marker='o')
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
