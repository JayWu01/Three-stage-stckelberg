import math
import numpy as np
import edge.experiment.Lag_Dual_Decompose.constants as cst
import matplotlib.pyplot as plt

# 用户任务数量
n_number = cst.n_number
error_value = cst.Error_value
# 初始化乘子 1*n_number的矩阵  （n_number代表用户数量）
lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
# 初始化 E_ij, E_ic
Eta_ij, Eta_ic = [], []
P_ij = cst.P_ij
P_ic = cst.P_ic
# 设置保留的小数位数
decimal_places = 5
Q_j, Q_c = cst.Q_j, cst.Q_c
a_e, a_c = cst.Alpha_e, cst.Alpha_c
phi_e, phi_c = cst.phi_e, cst.phi_c
D_i = cst.D_i
Epsilon_1 = cst.Epsilon_1
t_i_max = cst.t_i_max
f_i, f_j, f_c = cst.f_i, cst.f_j, cst.f_c
R_ij, R_ic = cst.R_ij, cst.R_ic
cpi = cst.cpi
v_p = cst.v_p
finshTaskCount = 0


def checkConstrain():
    global finshTaskCount
    finshTaskCount = 0
    for i in range(n_number):
        g1 = -Eta_ij[i] <= 0
        g2 = -Eta_ic[i] <= 0
        g3 = Eta_ij[i] + Eta_ic[i] - 1 <= 0
        g4 = 1 - Eta_ij[i] - Eta_ic[i] - (t_i_max[i] * f_i / (D_i[i] * cpi)) <= 0
        g5 = Eta_ij[i] - (t_i_max[i] / (D_i[i] * (cpi * (f_j ** -1) + R_ij ** -1))) <= 0
        g6 = Eta_ic[i] - (t_i_max[i] / (D_i[i] * (cpi * (f_c ** -1) + R_ic ** -1))) <= 0
        sum_data = (sum([Eta_ij[index] * cst.D_i[index] for index in range(n_number)]))
        g7 = cst.Q_total - sum_data <= 0
        print("--------------------------超载：{}".format(sum_data - cst.Q_total))
        g8 = cst.Q_j - P_ij[i] <= 0
        g9 = cst.Q_c - P_ic[i] <= 0
        if g4 == True and g5 == True and g6 == True:
            finshTaskCount += 1
    return g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9


def caculate_Psi_Rho():
    Psi = np.array(
        [Q_j * D_i[i] - lamda_1[i] + lamda_3[i] - lamda_4[i] + lamda_5[i] for i in
         range(n_number)])  # 去除过载约束 lamda_7=0
    Rho = np.array([Q_c * D_i[i] - lamda_2[i] + lamda_3[i] - lamda_4[i] + lamda_6[i] for i in range(n_number)])
    return Psi, Rho


def caculate_PriceByLamda():
    global P_ij, P_ic, Eta_ij, Eta_ic
    Psi, Rho = caculate_Psi_Rho()
    p_ij_max, p_ij_min, p_ic_max, p_ic_min = cst.p_ij_max, cst.p_ij_min, cst.p_ic_max, cst.p_ic_min
    for i in range(n_number):
        if Psi[i] / (D_i[i] - phi_e * lamda_8[i]) >= 0 and Psi[i] >= 0:
            # P_ij[i] = math.sqrt(a_e * phi_e * Psi[i] / (Epsilon_1 * (D_i[i] - phi_e * lamda_8[i])))
            P_ij[i] = np.maximum(Q_j, math.sqrt(a_e * phi_e * Psi[i] / (Epsilon_1 * (D_i[i] - phi_e * lamda_8[i]))))
        else:
            # P_ij[i] = p_ij_min
            P_ij[i] = Q_j
        if Rho[i] / (D_i[i] - phi_c * lamda_9[i]) >= 0 and Rho[i] >= 0:
            # P_ic[i] = math.sqrt(a_c * phi_c * Rho[i] / (Epsilon_1 * (D_i[i] - phi_c * lamda_9[i])))
            P_ic[i] = np.maximum(Q_c, math.sqrt(a_c * phi_c * Rho[i] / (Epsilon_1 * (D_i[i] - phi_c * lamda_9[i]))))
        else:
            # P_ic[i] = cst.p_ic_min
            P_ic[i] = Q_c
        Eta_ij[i] = (cst.Alpha_e / (cst.Epsilon_1 * P_ij[i])) - (1 / cst.phi_e)
        Eta_ic[i] = (cst.Alpha_c / (cst.Epsilon_1 * P_ic[i])) - (1 / cst.phi_c)
        pij_v.append(P_ij[i])
        pic_v.append(P_ic[i])
        Eta_ij_v.append(Eta_ij[i])
        Eta_ic_v.append(Eta_ic[i])
    return P_ij, P_ic


def caculate_gradient(P_ij, P_ic):
    lamda_1_grad = [-Eta_ij[i] for i in range(n_number)]
    lamda_2_grad = [-Eta_ic[i] for i in range(n_number)]
    lamda_3_grad = [Eta_ij[i] + Eta_ic[i] - 1 for i in range(n_number)]
    lamda_4_grad = [1 - Eta_ij[i] - Eta_ic[i] - (t_i_max[i] * f_i / (cpi * D_i[i])) for i in range(n_number)]
    lamda_5_grad = [Eta_ij[i] - (t_i_max[i] / (D_i[i] * (cpi * (f_j ** -1) + R_ij ** -1))) for i in range(n_number)]
    lamda_6_grad = [Eta_ic[i] - (t_i_max[i] / (D_i[i] * (cpi * (f_c ** -1) + R_ic ** -1))) for i in range(n_number)]
    lamda_7_grad = [0 for j in range(n_number)]  # 去除过载约束 lamda_7=0
    lamda_8_grad = [cst.Q_j - P_ij[i] for i in range(n_number)]
    lamda_9_grad = [cst.Q_c - P_ic[i] for i in range(n_number)]
    return lamda_1_grad, lamda_2_grad, lamda_3_grad, lamda_4_grad, lamda_5_grad, lamda_6_grad, lamda_7_grad, lamda_8_grad, lamda_9_grad


# 次梯度下降更新theta
Eta_ij_v, Eta_ic_v, pij_v, pic_v, L_v, f_v, U_d_v = [], [], [], [], [], [], []
lamda_2_v, lamda_3_v, Mu_1_v, Mu_2_v, Mu_3_v, Mu_4_v, Mu_5_v, Mu_6_v, Phi_v = [], [], [], [], [], [], [], [], []
U_CM_total_v, U_lag_dual_total_v = [], []


def update_lag_multipler():
    k = 0
    P_ij_v, P_ic_v, U_CM_v = [], [], []
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    print("初始化的乘子为：", lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9)
    for n in range(cst.max_iteration):
        print("价格P_ij为：", P_ij)
        print("价格P_ic为：", P_ic)
        P_ij_upd, P_ic_upd = caculate_PriceByLamda()
        print("第{}次边缘服务器定价决策为：{}".format(n + 1, P_ij_upd))
        print("第{}次云服务器定价决策为：{}".format(n + 1, P_ic_upd))
        print("第{}次边缘服务器卸载决策为：{}".format(n + 1, Eta_ij))
        print("第{}次云服务器卸载决策为：{}".format(n + 1, Eta_ic))
        lamda_1_grad, lamda_2_grad, lamda_3_grad, lamda_4_grad, lamda_5_grad, lamda_6_grad, lamda_7_grad, lamda_8_grad, lamda_9_grad = caculate_gradient(
            P_ij_upd, P_ic_upd)
        lamda_1 = [np.maximum(0, lamda_1[i] + cst.s_k * lamda_1_grad[i]) for i in range(n_number)]
        lamda_2 = [np.maximum(0, lamda_2[i] + cst.s_k * lamda_2_grad[i]) for i in range(n_number)]
        lamda_3 = [np.maximum(0, lamda_3[i] + cst.s_k * lamda_3_grad[i]) for i in range(n_number)]
        lamda_4 = [np.maximum(0, lamda_4[i] + cst.s_k * lamda_4_grad[i]) for i in range(n_number)]
        lamda_5 = [np.maximum(0, lamda_5[i] + cst.s_k * lamda_5_grad[i]) for i in range(n_number)]
        lamda_6 = [np.maximum(0, lamda_6[i] + cst.s_k * lamda_6_grad[i]) for i in range(n_number)]
        lamda_7 = [np.maximum(0, lamda_7[i] + cst.s_k * lamda_7_grad[i]) for i in range(n_number)]
        lamda_8 = [np.maximum(0, lamda_8[i] + cst.s_k * lamda_8_grad[i]) for i in range(n_number)]
        lamda_9 = [np.maximum(0, lamda_9[i] + cst.s_k * lamda_9_grad[i]) for i in range(n_number)]
        print("第{}次迭代更新的乘子为：".format(n + 1), lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8,
              lamda_9)
        # 终止条件判断
        U_M, U_C, U_v, U_lag_dual, U_D = caculate_L_f()
        U_CM = U_M + U_C + U_v
        f_v.append(-U_CM)
        L_v.append(-U_lag_dual)
        U_d_v.append(U_D)
        P_ij_v.append(P_ij_upd[0])
        P_ic_v.append(P_ic_upd[0])
        U_CM_v.append(-U_CM)
        print("第{}次迭代更新的效益函数值为：{}、对偶函数值为：{}".format(n + 1, U_CM, U_lag_dual))
        # 终止条件判断
        if ((k > 2) and (abs(U_CM - U_lag_dual) < error_value)) or k == cst.max_iteration - 1:
            break
        k += 1
    return U_M, U_C, U_v, U_lag_dual, U_D


# 注意这里的到的结果是负号的原因是原函数是先取了负号然后最小化的，但原函数原来是最大化
def caculate_L_f():
    lamda_1_grad, lamda_2_grad, lamda_3_grad, lamda_4_grad, lamda_5_grad, lamda_6_grad, lamda_7_grad, lamda_8_grad, lamda_9_grad = caculate_gradient(
        P_ij, P_ic)
    Q_total = cst.Q_total
    U_M = sum([Eta_ij[i] * (-P_ij[i]) * cst.D_i[i] for i in range(n_number)])
    for i in range(n_number):
        if Q_total < Eta_ij[i] * D_i[i]:
            break
        U_M += Eta_ij[i] * cst.Q_j * cst.D_i[i]
        Q_total -= Eta_ij[i] * D_i[i]
    # U_M = sum([Eta_ij[i] * (cst.Q_j - P_ij[i]) * cst.D_i[i] for i in range(n_number)])
    U_C = sum([Eta_ic[i] * (cst.Q_c - P_ic[i]) * cst.D_i[i] for i in range(n_number)])
    # U_v = v_p * (sum([Eta_ij[index] * cst.D_i[index]for index in range(n_number)])-cst.Q_total)
    U_v = -v_p * sum([Eta_ij[index] * cst.D_i[index] - cst.Q_total * (1 / n_number) for index in range(n_number)])
    U_lamda_1 = sum([lamda_1_grad[i] * lamda_1[i] for i in range(n_number)])
    U_lamda_2 = sum([lamda_2_grad[i] * lamda_2[i] for i in range(n_number)])
    U_lamda_3 = sum([lamda_3_grad[i] * lamda_3[i] for i in range(n_number)])
    U_lamda_4 = sum([lamda_4_grad[i] * lamda_4[i] for i in range(n_number)])
    U_lamda_5 = sum([lamda_5_grad[i] * lamda_5[i] for i in range(n_number)])
    U_lamda_6 = sum([lamda_6_grad[i] * lamda_6[i] for i in range(n_number)])
    U_lamda_7 = sum([lamda_7_grad[i] * lamda_7[i] for i in range(n_number)])
    # U_lamda_7 = 0  # 去除过载约束 lamda_7=0
    U_lamda_8 = sum([lamda_8_grad[i] * lamda_8[i] for i in range(n_number)])
    U_lamda_9 = sum([lamda_9_grad[i] * lamda_9[i] for i in range(n_number)])
    U_lag_dual = U_M + U_C + U_v + U_lamda_1 + U_lamda_2 + U_lamda_3 + U_lamda_4 + U_lamda_5 + U_lamda_6 + U_lamda_7 + U_lamda_8 + U_lamda_9

    U_d = sum([cst.Alpha_e * math.log(1 + cst.phi_e * Eta_ij[index]) * cst.D_i[index] + cst.Alpha_c * math.log(
        1 + cst.phi_c * Eta_ic[index]) * cst.D_i[
                   index] - cst.Epsilon_1 * (Eta_ij[index] * P_ij[index] + Eta_ic[index] * P_ic[index]) * cst.D_i[index]
               for
               index, e in
               enumerate(Eta_ij)])
    return U_M, U_C, U_v, U_lag_dual, U_d


def caculate_EtaByPrice():
    Eta_ij_init = [cst.Alpha_e / (cst.Epsilon_1 * element) - (1 / cst.phi_e) for element in P_ij]
    Eta_ic_init = [(cst.Alpha_c / (cst.Epsilon_1 * element) - (1 / cst.phi_c)) for element in P_ic]
    return Eta_ij_init, Eta_ic_init


def compared(n, l_th):
    global n_number, v_p
    global Eta_ij, P_ij, Eta_ic, P_ic
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    v_p = (Q_j - cst.vp_min) * np.exp(-0.1 * l_th) + cst.vp_min
    # v_p = 0.5 ** (1 / l_th) * Q_j
    n_number = n
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    U_CM = U_M + U_C + U_v
    return U_CM


def comparedTaskUtility(n):
    global n_number
    global Eta_ij, P_ij, Eta_ic, P_ic
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    n_number = n
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    U_CM = U_M + U_C + U_v
    U_d = [cst.Alpha_e * math.log(1 + cst.phi_e * Eta_ij[index]) * cst.D_i[index] + cst.Alpha_c * math.log(
        1 + cst.phi_c * Eta_ic[index]) * cst.D_i[
               index] - cst.Epsilon_1 * (Eta_ij[index] * P_ij[index] + Eta_ic[index] * P_ic[index]) * cst.D_i[index]
           for
           index, e in
           enumerate(Eta_ij)]
    return U_d, Eta_ij, Eta_ic


def comparedTaskCostAndEnergy(n):
    global n_number
    global Eta_ij, P_ij, Eta_ic, P_ic
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    n_number = n
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    return Eta_ij, Eta_ic, P_ij, P_ic


def compared_Sk(s_k):
    global Eta_ij, P_ij, Eta_ic, P_ic
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
    cst.s_k = s_k
    global f_v, L_v
    f_v, L_v = [], []
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    return U_M + U_C + U_v, f_v, L_v


def comparedEdgeCost(Qj):
    global Eta_ij, P_ij, Eta_ic, P_ic
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    cst.Q_j = Qj
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    return U_M + U_C + U_v


def comparedTaskFinshRate(n):
    global n_number
    global Eta_ij, P_ij, Eta_ic, P_ic
    global lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9
    n_number = n
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = cst.lamda_1, cst.lamda_2, cst.lamda_3, cst.lamda_4, cst.lamda_5, cst.lamda_6, cst.lamda_7, cst.lamda_8, cst.lamda_9
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    checkConstrain()
    return U_M, finshTaskCount




if __name__ == '__main__':
    cst.Task.read(n_number)
    cst.Task.readPrice(n_number)
    cst.Lamd.read(n_number)
    Eta_ij, Eta_ic = caculate_EtaByPrice()
    # 更新拉格朗日乘子
    U_M, U_C, U_v, U_lag_dual, U_D = update_lag_multipler()
    U_CM = U_M + U_C + U_v
    # 检验纳什均衡解是否满足约束
    checkConstrain()
    caculate_L_f()
    print("领导者效益函数值：", -U_CM)
    print("追随者效益函数值：", U_D)
    print("边缘服务器的效益为：", U_M)
    print("领导者对偶函数值：", -U_lag_dual)
    print("领导者对偶间隙：", abs(U_CM - U_lag_dual))
    print("边缘服务器的定价为：", P_ij)
    print("云服务器的定价为：", P_ic)
    print("边缘服务器的卸载决策为：", Eta_ij)
    print("云服务器的卸载决策为：", Eta_ic)
    print("卸载总比例为：", np.array(Eta_ij) + np.array(Eta_ic))
    print("边缘服务器的定价阈值为：", cst.p_ij_min, cst.p_ij_max)
    print("云服务器的定价阈值为：", cst.p_ic_min, cst.p_ic_max)


    # plt.figure(figsize=(10,6))
    plt.figure(1, constrained_layout=True)
    # plt.plot(range(2, len(f_v) + 1), np.array(f_v[1:]) - np.array(L_v[1:]), '-b', linewidth=2.0, label="Dual gap")
    plt.plot(range(2, len(f_v) + 1), abs(np.array(f_v[1:]) - np.array(L_v[1:])), '-b', linewidth=2.0)
    # 画y=0的渐近线
    plt.axhline(y=0, color='red', linestyle='--')
    plt.legend()
    plt.xlabel('iterations', fontsize=16, fontweight='bold')
    plt.ylabel('Duality gap value', fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=16)  # 修改x轴刻度字体大小
    plt.yticks(fontsize=16)  # 修改y轴刻度字体大小

    plt.show()


    plt.figure(2, constrained_layout=True)
    plt.plot(range(2, len(f_v) + 1), f_v[1:], '-b', linewidth=2.0, label="Leader Utilty")
    plt.plot(range(2, len(U_d_v) + 1), U_d_v[1:], '-g', linewidth=2.0, label="Follower Utilty")
    # 在某个点上添加坐标标签
    point_index = 79  # 你要在哪个点上添加标签，这里选择索引2的点
    plt.scatter(range(2, len(f_v) + 1)[point_index], f_v[point_index], marker='o', color='red')
    plt.text(range(2, len(f_v) + 1)[point_index], f_v[point_index], f'{f_v[point_index]:.4f}', ha='right', fontsize=12,
             color='red', weight='bold', va='bottom')

    plt.scatter(range(2, len(U_d_v) + 1)[point_index], U_d_v[point_index], marker='o', color='black')
    plt.text(range(2, len(U_d_v) + 1)[point_index], U_d_v[point_index], f'{U_d_v[point_index]:.4f}', ha='right',
             fontsize=12, color='red', weight='bold', va='bottom')

    # 添加图例并设置字体大小
    plt.legend(fontsize='16')
    plt.xlabel('iterations', fontsize=16, fontweight='bold')
    plt.ylabel('Utility value', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)  # 修改x轴刻度字体大小
    plt.yticks(fontsize=16)  # 修改y轴刻度字体大小
    
    plt.tight_layout()
    plt.show()
