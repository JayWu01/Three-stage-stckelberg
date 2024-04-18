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
                  [32.791809406956254, 61.45547834376258, 87.81558649881225, 110.98826920154664, 129.985122405639,
                  152.94478564920456, 172.15168158998304, 197.1460774274373, 224.06376880508847, 239.07358797112425],
                  [33.01216559895422, 62.011522327394864, 88.779494720962, 112.5171348758729, 131.8932351369496,
                  155.33950643220672, 175.02826714008256, 200.50073042411702, 227.99551997589728, 243.51263572812726]]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(5, 55, 5), socialWelfare[0], label='v=0', marker='.')
plt.plot(range(5, 55, 5), socialWelfare[1], label='v=1', marker='o')
plt.plot(range(5, 55, 5), socialWelfare[2], label='v=2', marker='^')
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
