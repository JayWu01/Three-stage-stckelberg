import matplotlib.pyplot as plt
# 预设字体格式，并传给rc方法
import numpy as np
# 四组数据
np.set_printoptions(precision=16)

U_user_v, U_S_t_v, U_vop_v = [4.446175506742655, 3.695959251718796, 3.3560239647087537, 3.1266285981296327,
                              2.810665575000767, 2.7129759519862087, 2.4179068202155336, 2.398926291315448,
                              2.3720723742150187, 2.311940957465355], [3.4356828992203714, 7.783528623816274,
                                                                       11.848873440756568, 14.749141407737547,
                                                                       17.82519758748793, 20.53470142991458,
                                                                       26.556771266340263, 29.75162665016549,
                                                                       34.15105599500961, 34.249211231658364], [
                                 0.8475742403409048, 2.144062817939837, 4.239640238488973, 6.066094557230833,
                                 7.486556260650842, 10.30827288531596, 11.049425908999307, 14.475436553734891,
                                 16.07045486722827, 19.22405029864413]

# U_S_t_v = [U_S_t_v[i] * (i + 1) for i in range(len(U_S_t_v))]
# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label='$U_{i}$', marker='*', color='#d95319')
plt.plot(range(len(U_S_t_v)), U_S_t_v, label='$U^{ecsp}_{j}$', marker='o')
plt.plot(range(len(U_vop_v)), U_vop_v, label='$U_{vop}$', marker='^', color='green')
# 添加图例
plt.legend()
plt.xlim(0, 9)
# 添加标题和轴标签
plt.xlabel('User Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Average Utility', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(U_user_v)), range(5, 55, 5), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
