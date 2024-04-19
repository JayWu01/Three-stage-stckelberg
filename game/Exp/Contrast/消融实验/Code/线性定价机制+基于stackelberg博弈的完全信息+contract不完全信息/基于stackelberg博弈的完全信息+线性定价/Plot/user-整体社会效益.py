import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [
    [33.487242176288525, 62.699335251304184, 90.5423243813811, 113.49689457861072, 132.08784397417617, 154.4092008940024, 176.5356471855541, 201.13081615418938, 227.00918058348435, 239.64841625674418],
    [33.79715358638257, 62.5725283997306, 91.80954622237273, 115.47927833727186, 134.70453505219447, 157.78235579340617, 177.35816889739533, 202.29905698318657, 232.31557675389396, 245.98314057279828],
    [30.70255582208295, 102.56158728588612, 103.29350763176198, 108.68585602066902, 104.91538033084652, 116.94841981498152, 122.42451916165233, 116.23140850494175, 99.36033014650889, 129.46188604691088]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(5, 55, 5), socialWelfare[0], label='基于合同的不完全信息', marker='^')
plt.plot(range(5, 55, 5), socialWelfare[1], label='基于stackelberg的完全信息', marker='.',color = 'red')
plt.plot(range(5, 55, 5), socialWelfare[2], label='线性定价', marker='o')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('user数量', fontsize=16)
plt.ylabel('整体社会效益', fontsize=16)
plt.xticks(range(5, 55, 5), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
