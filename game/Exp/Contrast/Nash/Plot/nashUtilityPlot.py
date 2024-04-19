import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
U_user_v, U_S_t_v, U_vop_v = [9.098345528870142, 7.868087828956858, 6.946495359721777, 6.228250148191359,
                              5.651727446387, 5.178217857419935, 4.782078899434659, 4.445603405192788,
                              4.156147185607261, 3.904428032957579, 3.7934886146592306, 3.5807617428836593,
                              3.4927334323649633, 3.4175807482494585, 3.3537841283415255, 3.1782054806455085,
                              3.1266285981296327, 3.1266285981296327], [
                                 [-110.87629191905702, -125.93730912797115, -62.29466205925412],
                                 [-62.50949432171879, -100.11861973277092, -34.65826744129379],
                                 [-35.25343659898995, -81.02175966723816, -18.312295357131823],
                                 [-18.60124892155651, -66.20703639558482, -8.049800728504152],
                                 [-7.871118566849335, -54.26663252844271, -1.353720093206902],
                                 [-0.7094059932129095, -44.3390677303393, 3.1149580776599555],
                                 [4.174875189336808, -35.87077759308153, 6.122103719377994],
                                 [7.544685008838902, -28.59098542181565, 8.068838795061728],
                                 [9.864665669257956, -22.12993985438, 9.326565462581918],
                                 [10.036491267674236, -12.612902783366849, 9.685995766228046],
                                 [11.690159154484896, -4.892662647577016, 10.71422093040325],
                                 [12.16669104024374, 0.6562934804274683, 11.226016118377594],
                                 [12.434304065774029, 6.650018426616356, 11.648399946172045],
                                 [12.578270525586339, 10.51373385732455, 12.0216679533432],
                                 [14.139123422959774, 13.970020288986877, 13.148703437009738],
                                 [14.904104376735697, 14.616891590965588, 13.843843655260795],
                                 [14.79009410070727, 15.65431129336546, 13.803018828452515],
                                 [14.79009410070727, 15.65431129336546, 13.803018828452515]], [78.57544115385171,
                                                                                               57.46170031136785,
                                                                                               45.75525805524922,
                                                                                               38.538901318151474,
                                                                                               33.749281705599586,
                                                                                               30.392412073673693,
                                                                                               27.939256400179715,
                                                                                               25.3614073048957,
                                                                                               23.389102227413463,
                                                                                               21.865619132295862,
                                                                                               16.944480808093402,
                                                                                               15.987086853955823,
                                                                                               12.435580527234809,
                                                                                               9.508316823446545,
                                                                                               7.6076098016644496,
                                                                                               7.296842178800844,
                                                                                               6.066094551440274,
                                                                                               6.066094551440274]

# 绘制折线图
# plt.plot(range(len(U_user_v)), U_user_v, label='用户的平均效益值', marker='.')
# plt.plot(range(len(U_S_t_v[:,0])), U_S_t_v[:,0], label='云服务器的效益值', marker='o')
# plt.plot(range(len(U_S_t_v[:,1])), U_S_t_v[:,1], label='M1服务器的效益值', marker='s')
# plt.plot(range(len(U_S_t_v[:,2])), U_S_t_v[:,2], label='M2服务器的效益值', marker='^')
# plt.plot(range(len(U_vop_v)), U_vop_v, label='vop的效益值', marker='x')
U_S_t_v = np.array(U_S_t_v)
# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label='The average utility value of users', marker='.')
plt.plot(range(len(U_S_t_v[:, 0])), U_S_t_v[:, 0], label='The utility value of $s_{1}$', marker='o')
plt.plot(range(len(U_S_t_v[:, 1])), U_S_t_v[:, 1], label='The utility value of $s_{2}$', marker='s')
plt.plot(range(len(U_S_t_v[:, 2])), U_S_t_v[:, 2], label='The utility value of $s_{3}$', marker='^')
plt.plot(range(len(U_vop_v)), U_vop_v, label='The utility value of VOP', marker='x')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Utility', fontsize=16)
plt.xticks(range(len(U_user_v)),fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
