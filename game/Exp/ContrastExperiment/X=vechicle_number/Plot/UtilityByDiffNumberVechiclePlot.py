import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size":16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据

U_user_v = [0.4758508100833219, 0.4842389558729726, 0.48872888779045487, 0.49342038520000153, 0.49342038520000153, 0.5021276254885036, 0.5021276254885036, 0.5160008347598913, 0.5160008347598913, 0.5274068951947001, 0.5366653185965735]
U_C_t_v = [2.6800000030205595, 2.67531535027658, 2.6932882305335504, 2.6872175980960753, 2.7064623994676342, 2.7180587717413403, 2.7356515946002684, 2.7077031167150025, 2.71850912844136, 2.7270463385675834, 2.7353105581927544]
U_M1_t_v = [5.546938775510203, 5.546938775510203, 5.546938775510203, 5.547746317739214, 5.552427251367979, 5.55452825621644, 5.565596700325253, 5.559368689437238, 5.5679859208001945, 5.574436887519391, 5.567869936764913]
U_M2_t_v = [7.11, 7.11, 7.11, 7.109999999999999, 7.109999999999999, 7.108805246913578, 7.108807011189673, 7.10655899592334, 7.107815445935952, 7.107115095676302, 7.109494668643192]
U_vop_v = [-4.2892891165246425, -2.1088843411138387, -0.479426459464943, 0.09380517798641602, 0.44395097948131057, 0.6840092456908836, 0.9377814704635329, 1.145833896999759, 1.1919415043468355, 1.7475578677417318, 2.088010337203335]


# 绘制折线图
plt.plot(range(len(U_user_v)), U_user_v, label='xx', marker='.')
plt.plot(range(len(U_C_t_v)), U_C_t_v, label='xx', marker='o')
plt.plot(range(len(U_M1_t_v)), U_M1_t_v, label='xx', marker='s')
plt.plot(range(len(U_M2_t_v)), U_M2_t_v, label='xx', marker='^')
plt.plot(range(len(U_vop_v)), U_vop_v, label='xx', marker='x',color='red')
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
