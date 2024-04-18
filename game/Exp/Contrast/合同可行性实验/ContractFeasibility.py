import matplotlib.pyplot as plt
import game.Config.constants as cst

# 预设字体格式，并传给rc方法
# font = {'family': 'SimHei', "size": 16}
# matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据
v_number = cst.v_number
cst.Vechicle.read(v_number)
Theta_m = cst.Theta_m
Q_total_m = cst.Q_total_m
f_m, p_m = [0.3663281316293854, 0.537053260594265, 0.5370793272004318, 0.8714450740138977, 0.8903638495297987,
            1.233993395268722, 1.4344773201995156, 1.480073315890596, 1.6514995133208286, 1.965446790673886], [
               0.05092982795217462, 0.1027251009857435, 0.10273418227905515, 0.24312830890975193, 0.25223830156038934,
               0.4354613102653315, 0.5603149701008263, 0.5897510252059383, 0.7005549029682445, 0.905631462924501]

V2, V4, V6, V8, V10 = [p_m[i] - (f_m[i] ** 2 / Theta_m[1]) for i in range(len(Theta_m))], [
    p_m[i] - (f_m[i] ** 2 / Theta_m[3])
    for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[5]) for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[7])
                          for i in range(len(Theta_m))], [
                          p_m[i] - (f_m[i] ** 2 / Theta_m[9]) for i in range(len(Theta_m))]

plt.plot(range(1, len(V2) + 1), V2, label=r'$\theta_{2}$', c='r', marker='.')
plt.axvline(x=2, ymin=0, ymax=0.65, color='r', linestyle='--')

plt.plot(range(1, len(V4) + 1), V4, label=r'$\theta_{4}$', c='black', marker='*')
plt.axvline(x=4, ymin=0, ymax=0.66, color='black', linestyle='--')

plt.plot(range(1, len(V6) + 1), V6, label=r'$\theta_{6}$', c='blue', marker='<')
plt.axvline(x=6, ymin=0, ymax=0.72, color='blue', linestyle='--')

plt.plot(range(1, len(V8) + 1), V8, label=r'$\theta_8$', c='y', marker='.')
plt.axvline(x=8, ymin=0, ymax=0.81, color='y', linestyle='--')

plt.plot(range(1, len(V10) + 1), V10, label=r'$\theta_{10}$', c='g', marker='^')
plt.axvline(x=10, ymin=0, ymax=0.95, color='g', linestyle='--')

# 调整部分刻度的间隔
plt.legend()
# 添加标题和轴标签
plt.xlabel(r'$\theta_m$', fontsize=16)
plt.ylabel('$U_{m}$', fontsize=16)
plt.xticks(range(1, len(V8) + 1, 1), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
