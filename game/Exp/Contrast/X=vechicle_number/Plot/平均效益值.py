import matplotlib.pyplot as plt

# 预设字体格式，并传给rc方法
import numpy as np
# 四组数据
np.set_printoptions(precision=16)

U_user_v, U_S_t_v, U_vop_v = [2.929951276545613, 2.929951276545613, 2.929951276545613, 2.9720430330062255,
                              2.9720430330062255, 2.9720430330062255, 2.9720430330062255, 3.1266285981296327,
                              3.1266285981296327, 3.1266285981296327, 3.1266285981296327, 3.1266285981296327,
                              3.1266285981296327], [16.221873263888895, 16.279658841556554, 16.33696730581911,
                                                    15.811259676825664, 15.876537376312772, 15.94479099708856,
                                                    15.988457519721393, 14.546639611908812, 14.626544413173328,
                                                    14.670629800378023, 14.728665061125332, 14.765426936992435,
                                                    14.89828143061464], [-7.38855274664109, -1.4413260998969244,
                                                                         0.7277221967345078, 1.8713940231281638,
                                                                         3.0284362826979394, 4.154984741980421,
                                                                         4.415784184824833, 5.129263839674659,
                                                                         6.094979940144398, 6.3774994682900825,
                                                                         6.066094557230833, 5.888465754416272,
                                                                         7.325860622702053]

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

plt.xlim(0, 12)

# 添加标题和轴标签
plt.xlabel('Vechicle Number', fontweight='bold', fontsize=15.5)
plt.ylabel('Average Utility', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(U_user_v)), range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
