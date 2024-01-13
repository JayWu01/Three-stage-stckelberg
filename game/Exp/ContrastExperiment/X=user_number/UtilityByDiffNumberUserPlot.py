import matplotlib.pyplot as plt
import matplotlib

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size": 16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据

U_C_t = [2.141622534769942, 2.7165050218061344, 2.970607362798039, 3.1385755854899697, 3.360297842668812,
         3.4421896736626056, 3.5574389158368795]
U_M1 = [3.84791409435387, 5.572737410729378, 6.65297595250083, 7.472489776635923, 8.088558238482555, 8.524538290512684,
        8.982054292382697]
U_M2 = [4.709129070607753, 7.104747879968234, 8.758313430563785, 10.009051063365156, 10.997727864736433,
        11.800168550426273, 12.488093643070632]
U_Vop = [0.7126600016906195, 1.1928068463221584, 1.601926878972256, 1.812577300993521, 1.8893727004645555,
         2.131764182543114, 2.0753694785711225]

# 绘制折线图
X_range = [10, 20, 30, 40, 50, 60, 70]

plt.plot(X_range, U_C_t, marker='o', color="black")
plt.plot(X_range, U_M1, marker='o', color="red")
plt.plot(X_range, U_M2, marker='s', color="blue")
plt.plot(X_range, U_Vop, marker='^', color="green")
# plt.plot(range(len(p_j_vop_1)), p_j_vop_1,marker='*')
# plt.plot(range(len(p_j_vop_2)), p_j_vop_2, marker='<')


# 添加标题和轴标签
plt.xlabel('用户数量', fontsize=16)
plt.ylabel('效益值', fontsize=16)
plt.xticks(fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
