import matplotlib.pyplot as plt
# 预设字体格式，并传给rc方法
import numpy as np
from numpy import array
# 四组数据
np.set_printoptions(precision=16)
fig = plt.subplots()
U_user_v, U_S_t_v = [array([0.4900980392156867]), array([1.271666666666666, 0.9224404761904752]),
                     array([1.1294074074074067, 0.8458080808080801, 0.9316845878136201]),
                     array([0.9373214285714286, 0.822791666666666, 0.7325862068965516,
                            0.4449621212121203]), array([0.9859444444444453, 0.6572962962962958, 0.7465066666666673,
                                                         0.4472631578947356, 0.4472631578947356]),
                     array([1.0478282828282837, 0.5799673202614372, 0.7848792270531411,
                            0.3625617283950612, 0.3625617283950612, 0.3625617283950612]),
                     array([0.7466233766233772, 0.6174702380952368, 0.4967701863354045,
                            0.3850700280112037, 0.3850700280112037, 0.3850700280112037,
                            0.3850700280112037]), array([0.8644583333333358, 0.5763055555555557, 0.5851984126984151,
                                                         0.3319531250000001, 0.3319531250000001, 0.3319531250000001,
                                                         0.3319531250000001, 0.3319531250000001]),
                     array([0.4535925925925941, 0.3023950617283946, 0.7406237816764142,
                            0.3023950617283946, 0.3023950617283946, 0.3023950617283946,
                            0.3023950617283946, 0.3023950617283946, 0.8617378917378915]),
                     array([0.6572962962962962, 0.3034999999999993, 0.3595438596491236,
                            0.3034999999999993, 0.3034999999999993, 0.3034999999999993,
                            0.3034999999999993, 0.3034999999999993, 0.9096388888888887,
                            0.3034999999999993])], [3.0121339187837775, 5.653112606582616, 3.773678957854967,
                                                    2.6785039737552196, 2.2147318757762067, 2.404215090646757,
                                                    1.508238789801278, 2.214135435490868, 1.4705678238384203,
                                                    1.3927451092886933]
U_user_v = [np.sum(arr) for arr in U_user_v]
# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均资源购买量', marker='.',color='green')
plt.plot(range(len(U_user_v)), U_user_v, label='$f_i$', marker='*', color='#d95319',linewidth=4,markersize=10)
plt.plot(range(len(U_S_t_v)), U_S_t_v, label='$f^{j}_{vop}$', marker='o',linewidth=4,markersize=10)

# 添加图例
plt.legend()
plt.xlim(0, 9)
# 添加标题和轴标签
plt.xlabel('ECSP Number', fontweight='bold', fontsize=20)
plt.ylabel('Average Resource Purchase', fontweight='bold', fontsize=20)
plt.xticks(range(len(U_S_t_v)), range(1, 11), fontsize=20)  # 修改x轴刻度字体大小
plt.yticks(fontsize=20)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize=17.5)
plt.subplots_adjust(bottom=0.14, left=0.12, right=0.965, top=0.97)
# 显示图形

# 保存图像时设置dpi参数
plt.savefig("Fig_5c.png", dpi=300)
plt.show()
