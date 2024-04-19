import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [[100.12796887802119, 106.19076668010068, 108.53256967558933, 109.1990348436584, 110.57933143826808,
                  111.92889329377228, 112.44745879798933, 111.80379317505611, 112.91576987801949, 113.48704372270964,
                  113.49689457861072, 113.68633508671272, 115.29664418564332],
                 [100.12796887802119, 106.19074056307332, 108.62293932900043, 109.49602946390058, 111.13651417199685,
                  112.46930597436842, 111.92044186675678, 112.83575817319243, 113.71838522341044, 114.5827439153567,
                  115.47927833727186, 116.38177135448406, 96.71941265932986],
                 [100.12796887802119, 103.72668668152207, 106.28626133938467, 109.32346354014297, 109.33956978766159,
                  100.08038357277965, 100.016890534361, 110.5087294225272, 100.1743315044348, 107.49552827799562,
                  106.04595464692619, 111.03448936778115, 105.2394387630879]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')
# plt.plot(range(1, 10), socialWelfare[0], label='基于stackelberg的完全信息', marker='.')
# plt.plot(range(1, 10), socialWelfare[1], label='线性定价-p=1.2', marker='o')
# plt.plot(range(1, 10), socialWelfare[2], label='线性定价-p=1.5', marker='*')
# plt.plot(range(1, 10), socialWelfare[3], label='基于合同的不完全信息', marker='^')

# 绘制折线图
plt.plot(range(0, 13), socialWelfare[0], label='基于合同的不完全信息', marker='^')
plt.plot(range(0, 13), socialWelfare[1], label='基于stackelberg的完全信息', marker='.')
plt.plot(range(0, 13), socialWelfare[2], label='线性定价', marker='o')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('车辆数量', fontsize=16)
plt.ylabel('整体社会效益', fontsize=16)
plt.xticks(range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
