import matplotlib.pyplot as plt
import matplotlib
import game.Config.constants as cst

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
v_number = cst.v_number
cst.Vechicle.read(v_number)
Theta_m = cst.Theta_m
Q_total_m = cst.Q_total_m

# f_m, p_m = [0.5018125670385615, 0.7355553293978258, 0.7357815681406069, 1.193744219639791, 1.2196599997482793, 1.6903790343213578, 1.9650108311533587, 2.02747025391909, 2.2622974832871985, 2.6923564265154276] , [0.09556849211255114, 0.1926996470595562, 0.19280761241645833, 0.4562243895351065, 0.4733190528352731, 0.8171322437751514, 1.0514168758853637, 1.1066528645633378, 1.3145734475252704, 1.6993942566796392]
f_m, p_m = [0.5018124994094384, 0.7356837532581453, 0.7357126892453469, 1.1937440587592552, 1.2196598353750818, 1.6903788065095031, 1.9650105663294624, 2.0274699806775565, 2.262297178398073, 2.692356063667392] , [0.09556846635308112, 0.1927630968533515, 0.1927769062061561, 0.45622378277578146, 0.47331844146826335, 0.8171315397369551, 1.051416108698238, 1.1066520824879393, 1.3145726094070918, 1.6993933148371063]
V2, V4, V6, V8, V10 = [p_m[i] - (f_m[i] ** 2 / Theta_m[1]) for i in range(len(Theta_m))], [
    p_m[i] - (f_m[i] ** 2 / Theta_m[3])
    for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[5]) for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[7])
                          for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[9]) for i in range(len(Theta_m))]

plt.plot(range(1, len(V2) + 1), V2, label='V2', c='r', marker='.')
plt.axvline(2, ymin=0.0, ymax=0.035 + 0.1, c='r', ls='--')

plt.plot(range(1, len(V4) + 1), V4, label='V4', c='black', marker='*')

# plt.axvline(4, ymin=0.0, ymax=0.65 + V4[3], c='black', ls='--')

plt.plot(range(1, len(V6) + 1), V6, label='V6', c='blue', marker='<')
# plt.axvline(6, ymin=0.0, ymax=0.65 + V6[-5], c='blue', ls='--')

plt.plot(range(1, len(V8) + 1), V8, label='V8', c='y', marker='.')
# plt.axvline(8, ymin=0.0, ymax=0.65 + V8[-3], c='y', ls='--')

plt.plot(range(1, len(V10) + 1), V10, label='V10', c='g', marker='^')
# plt.axvline(10, ymin=-0.4, ymax=0.65 + V10[-1], c='g', ls='--')

# 调整部分刻度的间隔
plt.legend()
# 添加标题和轴标签
plt.xlabel('xx', fontsize=16)
plt.ylabel('xx', fontsize=16)
plt.xticks(range(1,len(V8),1),fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
