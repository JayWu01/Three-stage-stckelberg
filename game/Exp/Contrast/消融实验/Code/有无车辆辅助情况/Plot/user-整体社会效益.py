import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [[25.811845755352152, 55.974015413717794, 79.11023384379108, 100.12796887802119, 116.73829392911496,
                  140.2743720037725, 151.9611471928926, 174.43581519114485, 203.40153162105543, 209.21416520313693],
                 [32.82512093443793, 61.53953599737898, 87.96130115044652, 111.21938871788777, 130.27359070340125,
                  153.30679666896066, 172.58653667672192, 197.6533811061776, 224.65813354375277, 239.74413850759305],
                 [33.094985406132, 62.22050898137436, 89.14177552184333, 113.09175271793909, 132.61042491114424,
                  156.23955243548681, 176.10941993041556, 201.7614690434735, 229.47325250949737, 245.18100861595607]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(5, 55, 5), socialWelfare[0], label='Without Vechicle', marker='.')
plt.plot(range(5, 55, 5), socialWelfare[1], label='The number of Vechicle is =5', marker='o')
plt.plot(range(5, 55, 5), socialWelfare[2], label='The number of Vechicle is =10', marker='^')

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
