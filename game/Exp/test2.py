import math
import numpy as np
# import edge.experiment.Lag_Dual_Decompose.constants as cst
import matplotlib.pyplot as plt

# 提供的数据
p_0_t_c,p_1_t_c,p_2_t_c,p_3_t_c=[7.660296430696015, 7.653974638112235, 7.650344259369426, 7.646705790204244, 7.643059230616698, 7.637817053341266, 7.628463994646044, 7.621099410073369, 7.620084934562762, 7.6163797194130485, 7.6126664138409685, 7.60894501784652, 7.605215531429707, 7.601477954590521, 7.597732287328971, 7.593978529645053, 7.590216681538767, 7.5928669690660335] , [11.516558240997227, 11.698475586222973, 11.795142762356955, 11.862432728458048, 11.89660826570628, 11.891209035842685, 11.837761747834644, 11.738042970277117, 11.60394823968089, 11.597986232512431, 11.592019508035614, 11.586048066250434, 11.580071907156897, 11.574091030755001, 11.568105437044744, 11.562115126026132, 11.556120097699157, 11.558137140131809] , [13.44966176592593, 13.75604672014568, 14.038411792261986, 14.311343861890064, 14.573690757888462, 14.821226990308544, 15.046855369499713, 15.257799543067541, 15.462926561139815, 15.647782677480228, 15.809850673706766, 15.945615577994577, 16.05086128328697, 16.120496171053453, 16.148326012567882, 16.126755117550033, 16.046388608606136, 15.933424260861724] , [4.264010655211038, 4.264010655211038, 4.433960668282975, 4.62375592775181, 4.8359927549773944, 5.073682094521717, 5.337908830087256, 5.627464074979431, 5.954050757552856, 6.025586352708701, 6.107767071432533, 6.202313728463739, 6.311294674266513, 6.437211828084298, 6.583112340056448, 6.75273493260377, 6.950703714990351, 7.182787806162116]
# p_0_t_c,p_1_t_c,p_2_t_c=[3.9, 4.0, 3.9, 3.8, 3.6999999999999997, 3.5999999999999996, 3.5999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996, 3.4999999999999996] , [3.8, 3.9, 3.8, 3.6999999999999997, 3.5999999999999996, 3.4999999999999996, 3.3999999999999995, 3.2999999999999994, 3.1999999999999993, 3.099999999999999, 2.999999999999999, 2.899999999999999, 2.799999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999, 2.699999999999999] , [3.75, 3.85, 3.75, 3.65, 3.55, 3.4499999999999997, 3.3499999999999996, 3.2499999999999996, 3.1499999999999995, 3.0499999999999994, 2.9499999999999993, 2.849999999999999, 2.749999999999999, 2.649999999999999, 2.549999999999999, 2.449999999999999, 2.3499999999999988, 2.2499999999999987, 2.1499999999999986, 2.0499999999999985, 1.9499999999999984, 1.9499999999999984, 1.8499999999999983, 1.8499999999999983]# 绘制折线图
plt.plot(p_0_t_c, label='p_0_t_c')
plt.plot(p_1_t_c, label='p_1_t_c')
plt.plot(p_2_t_c,label='p_2_t_c')
plt.plot(p_3_t_c,label='p_3_t_c')

# 添加标题和标签
plt.title('Line Chart of p_0_t_c, p_1_t_c, p_2_t_c')
plt.xlabel('Time')
plt.ylabel('Values')

# 添加图例
plt.legend()

# 显示图形
plt.show()

# plt.figure(figsize=(10,6))
# plt.figure(1, constrained_layout=True)
# # plt.plot(range(2, len(f_v) + 1), np.array(f_v[1:]) - np.array(L_v[1:]), '-b', linewidth=2.0, label="Dual gap")
# plt.plot(range(2, len(p_0_t_c) + 1), abs(np.array(p_0_t_c[1:]) - np.array(p_0_t_c[1:])), '-b', linewidth=2.0)
# # 画y=0的渐近线
# plt.axhline(y=0, color='red', linestyle='--')
# plt.legend()
# plt.xlabel('iterations', fontsize=16, fontweight='bold')
# plt.ylabel('Duality gap value', fontsize=16, fontweight='bold')
# plt.grid(True)
# plt.xticks(fontsize=16)  # 修改x轴刻度字体大小
# plt.yticks(fontsize=16)  # 修改y轴刻度字体大小
#
# plt.show()