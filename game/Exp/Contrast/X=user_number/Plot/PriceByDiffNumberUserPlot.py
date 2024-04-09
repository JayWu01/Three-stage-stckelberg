import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size":16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据

P_0_v = [4.6999999999999975, 5.399999999999995, 5.749999999999994, 5.999999999999993, 6.199999999999992, 6.349999999999992, 6.449999999999991, 6.549999999999991, 6.599999999999991, 6.69999999999999]
P_1_v = [3.85, 4.499999999999998, 4.849999999999997, 5.149999999999996, 5.349999999999995, 5.5499999999999945, 5.649999999999994, 5.799999999999994, 5.899999999999993, 5.999999999999993]
P_2_v = [3.5000000000000013, 4.1499999999999995, 4.499999999999998, 4.799999999999997, 4.9999999999999964, 5.149999999999996, 5.299999999999995, 5.449999999999995, 5.5499999999999945, 5.6499999999999945]
p_j_vop_v = np.array([
    [2.517926422840331, 1.678744328339085, 1.462120951715708],
    [3.431522475471561, 2.35364085553386, 2.0535605342487186],
    [3.9643248234966295, 2.817031123407888, 2.50519438871401],
    [4.373766866663001, 3.202067837536775, 2.8174335333296554],
    [4.8175818167722735, 3.5342446843355146, 3.124898889942987],
    [5.034488028761098, 3.7140247082977567, 3.304378138651184],
    [5.328026406314095, 3.9291777221035478, 3.5782740478235073],
    [5.483151649160861, 4.135671543060038, 3.7167155164861647],
    [5.53939181470748, 4.298251598990963, 3.881357416673049],
    [5.752646808471435, 4.418060342305983, 4.013635563544916]
])


# P_0_v,P_1_v,P_2_v = [0.30000000000000004, 0.4, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999, 1.0999999999999999, 1.2, 1.3, 1.4000000000000001, 1.5000000000000002, 1.6000000000000003, 1.7000000000000004, 1.8000000000000005, 1.8000000000000005, 1.8000000000000005, 1.9000000000000006, 1.9000000000000006, 1.9000000000000006, 2.0000000000000004, 2.0000000000000004, 2.0000000000000004, 2.1000000000000005, 2.1000000000000005, 2.1000000000000005, 2.2000000000000006, 2.2000000000000006, 2.2000000000000006, 2.2000000000000006, 2.2000000000000006, 2.2000000000000006, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.3000000000000007, 2.400000000000001, 2.400000000000001, 2.400000000000001, 2.400000000000001, 2.500000000000001, 2.500000000000001, 2.500000000000001, 2.500000000000001] , [0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999, 1.0999999999999999, 1.2, 1.3, 1.4000000000000001, 1.5000000000000002, 1.6000000000000003, 1.7000000000000004, 1.8000000000000005, 1.9000000000000006, 2.0000000000000004, 2.1000000000000005, 2.2000000000000006, 2.3000000000000007, 2.400000000000001, 2.500000000000001, 2.600000000000001, 2.700000000000001, 2.800000000000001, 2.9000000000000012, 3.0000000000000013, 3.1000000000000014, 3.2000000000000015, 3.3000000000000016, 3.4000000000000017, 3.5000000000000018, 3.600000000000002, 3.700000000000002, 3.800000000000002, 3.900000000000002, 4.000000000000002, 4.100000000000001, 4.200000000000001, 4.300000000000001, 4.4, 4.5, 4.6, 4.699999999999999, 4.799999999999999, 4.899999999999999, 4.999999999999998, 5.099999999999998, 5.099999999999998, 5.099999999999998, 5.099999999999998] , [0.44999999999999996, 0.5499999999999999, 0.6499999999999999, 0.7499999999999999, 0.8499999999999999, 0.9499999999999998, 1.0499999999999998, 1.15, 1.25, 1.35, 1.4500000000000002, 1.5500000000000003, 1.6500000000000004, 1.7500000000000004, 1.8500000000000005, 1.9500000000000006, 2.0500000000000007, 2.150000000000001, 2.250000000000001, 2.350000000000001, 2.450000000000001, 2.550000000000001, 2.6500000000000012, 2.7500000000000013, 2.8500000000000014, 2.9500000000000015, 3.0500000000000016, 3.1500000000000017, 3.2500000000000018, 3.350000000000002, 3.350000000000002, 3.350000000000002, 3.350000000000002, 3.450000000000002, 3.450000000000002, 3.450000000000002, 3.450000000000002, 3.450000000000002, 3.450000000000002, 3.450000000000002, 3.550000000000002, 3.550000000000002, 3.550000000000002, 3.550000000000002, 3.650000000000002, 3.650000000000002, 3.650000000000002, 3.550000000000002, 3.550000000000002]


p_0_vop_v,p_1_vop_v,p_2_vop_v=p_j_vop_v[:, 0],p_j_vop_v[:, 1],p_j_vop_v[:, 2]

# 绘制折线图
plt.plot(range(len(P_0_v)), P_0_v, label='xx')
plt.plot(range(len(P_1_v)), P_1_v, label='xx')
plt.plot(range(len(P_2_v)), P_2_v, label='xx')
# plt.plot(range(len(p_0_vop_v)), p_0_vop_v, label='xx', marker='^')
# plt.plot(range(len(p_1_vop_v)), p_1_vop_v, label='xx', marker='x')
# plt.plot(range(len(p_2_vop_v)), p_2_vop_v, label='xx', marker='x')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('xx', fontsize=16)
plt.ylabel('xx', fontsize=16)
plt.xticks(fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
