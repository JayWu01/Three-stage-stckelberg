import game.Config.constants as cst
import random

import numpy as np

nuser = cst.n_number
bg = cst.bg
Theta_m = cst.Theta_m
# 权重参数
alpha, beta, zeta = np.array([3.0, 3.0, 3.0]), np.array([5.0, 5.0, 5.0]), np.array([2.0, 2.0, 2.0])
e, a = [1.5, 1.0, 0.8], 1.0  # zuiyou
C = [0.5, 0.3, 0.2]  # 三台服务器成本
K = [2.0, 1.0, 0.8]  # 服务器功率

# 车辆类型为theta_m的概率
lamda_m = [0.04, 0.17, 0.09, 0.19, 0.01, 0.14, 0.02, 0.13, 0.01, 0.2]
v_number = cst.v_number
Q_total_m = cst.Q_total_m  # 车辆m的计算资源负载

# vop运营商自身的计算资源容量
Q_vop = 0
# CEA的计算资源上限
Q_CEA = [float("inf"), 12, 10]


# beta_v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# alpha_v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def create():
    global lamda_m
    # 生成10个均匀分布的概率值（小数点后最多两位）
    probabilities = [round(random.uniform(0, 1), 2) for _ in range(v_number)]

    # 确保概率值之和为1
    total_probability = sum(probabilities)

    # 计算归一化后的概率值，并保留小数点后两位
    lamda_m = [round(prob / total_probability, 2) for prob in probabilities]


def find_Optial_mulitUser(P_0, P_1, P_2):
    F_i0, F_i1, F_i2 = [], [], []
    for i in range(nuser):
        result = optimal_Stage3strategy_KKT(bg[i], P_0, P_1, P_2)
        F_i0 = np.append(F_i0, result[1])
        F_i1 = np.append(F_i1, result[2])
        F_i2 = np.append(F_i2, result[3])

    return F_i0, F_i1, F_i2


# 计算用户的效益函数值
def calculate_utility_for_user_device(F_i0, F_i1, F_i2, p_0_t, p_1_t, p_2_t):
    # fix(f_vop_0,P_vop)
    F = [F_i0, F_i1, F_i2]
    P = [p_0_t, p_1_t, p_2_t]
    U = [sum([alpha[j] * np.log(1 + beta[j] * F[j][i]) - zeta[j] * P[j] * F[j][i] for j in range(len(K))]) for i in
         range(nuser)]
    return U


# 计算云服务器的效益函数值
def calculate_utility_for_Cloud_server(p_0_t, p_1_t, p_2_t, p_vop_0, f_vop_0):
    # fix(f_vop_0,P_vop)
    F_i0, F_i1, F_i2 = find_Optial_mulitUser(p_0_t, p_1_t, p_2_t)

    # 奖励回报
    # Reward = p_0_t * np.log(1 + sum(F_i0))
    Reward = (p_0_t - C[0]) * sum(F_i0)
    # Reward = (np.log(1+p_0_t-C[0]))* sum(F_i0)
    # 计算 能耗、成本
    E = a * e[0] * K[0] * ((sum(F_i0) - f_vop_0) ** 2)
    # payment = p_vop_0 * np.log(1 + f_vop_0)
    payment = p_vop_0 * f_vop_0
    # 计算整体表达式
    U = Reward - E - payment
    return U


def calculate_utility_for_M1_server(p_1_t, p_0_t, p_2_t, p_vop_1, f_vop_1):
    # fix(f_vop_0,P_vop)
    # f_vop_1 = P_vop = 0.5
    F_i0, F_i1, F_i2 = find_Optial_mulitUser(p_0_t, p_1_t, p_2_t)

    # 奖励回报
    # Reward = p_1_t * np.log(1 + sum(F_i1))
    Reward = (p_1_t - C[1]) * sum(F_i1)
    # Reward = (np.log(1 + p_1_t - C[1])) * sum(F_i1)
    # 计算 能耗、成本
    E = a * e[1] * K[1] * ((sum(F_i1) - f_vop_1) ** 2)
    # payment = p_vop_1 * np.log(1 + f_vop_1)
    payment = p_vop_1 * f_vop_1
    # 计算整体表达式
    U = Reward - E - payment
    return U


def calculate_utility_for_M2_server(p_2_t, p_0_t, p_1_t, p_vop_2, f_vop_2):
    # fix(f_vop_0,P_vop)
    # f_vop_2 = P_vop = 0.5
    F_i0, F_i1, F_i2 = find_Optial_mulitUser(p_0_t, p_1_t, p_2_t)

    # 奖励回报
    # Reward = p_2_t * np.log(1 + sum(F_i2))
    Reward = (p_2_t - C[2]) * sum(F_i2)
    # Reward = (np.log(1 + p_2_t - C[2])) * sum(F_i2)
    # 计算 能耗、成本
    E = a * e[2] * K[2] * ((sum(F_i2) - f_vop_2) ** 2)
    # payment = p_vop_2 * np.log(1 + f_vop_2)
    payment = p_vop_2 * f_vop_2
    # 计算整体表达式
    U = Reward - E - payment
    return U


# stage I 计算Vop的效益
def calculate_utility_for_Vop(f_m, p_m, p_j_vop, F):
    # 回报
    Reward = sum([p_j_vop[i] * (sum(F[i]) - p_j_vop[i] / (2 * a * e[i] * K[i])) for i in range(len(K))])
    # 付给车辆的成本
    # payment_cost = v_number * sum([lamda_m[i] * (f_m[i] ** 2 / Theta_m[i] + sum(
    #     [(Theta_m[j - 1] ** -1 - Theta_m[j] ** -1) * f_m[j - 1] ** 2 for j in range(1, v_number)])) for i in
    #                                range(v_number)])

    payment_cost = sum([f_m[m] * p_m[m] for m in range(v_number)])

    # 计算整体表达式
    U_vop = Reward - payment_cost
    return U_vop


# 计算f_m梯度 
def caculate_VopGradient(f_m, p_m, p_j_vop, F, f_j_vop):
    Phi_m_grad = f_m
    Omega_m_grad = [Q_total_m[i] - f_m[i] for i in range(v_number)]
    Pi_grad = sum(f_m) - sum(f_j_vop)
    Upsilon_j_grad = p_j_vop
    Q_CEA[0] = sum(F[0])
    Lambda_j_grad = [2 * a * e[j] * K[j] * Q_CEA[j] - p_j_vop[j] for j in range(len(K))]
    return Phi_m_grad, Omega_m_grad, Pi_grad, Upsilon_j_grad, Lambda_j_grad


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def checkConstrain(f_m, p_m, p_j_vop, F):
    for j in range(v_number):
        g3 = f_m[j] >= 0
        g4 = Q_total_m[j] - f_m[j]
        g5 = v_number * sum([lamda_m[i] * f_m[i] for i in range(v_number)]) - sum(
            [sum(F[j]) - p_j_vop[j] / (2 * a * e[j] * K[j]) for j in range(len(K))])

    for j in range(len(K)):
        g1 = p_j_vop[j]
        g2 = sum(F[j]) * a * 2 * e[j] * K[j] - p_j_vop[j]
    return g1 >= 0 and g2 >= 0


p_j_vop_t, p_m_t, f_m_t = [], [], []

# vop自带服务器计算资源大小
# a_vop, e_vop, k_vop = 1.0, 1.0, 1.0
a_vop, e_vop, k_vop = 0.1, 0.1, 0.1
e_vk = 2 * e_vop * a_vop * k_vop


# 核心代码：拉格朗日交替更新拉格朗日乘子 stageI
def LagrangeDualStageIforVop(F):
    global p_j_vop_t, p_m_t, f_m_t
    Omega_m = cst.Omega_m  # 约束C1
    Phi_m = cst.Phi_m  # 约束C1
    Pi = 1.0  # 约束C2
    Upsilon_j = [1.0, 1.0, 1.0]  # #约束C3
    Lambda_j = [1.0, 1.0, 1.0]  # #约束C3
    # p_j_vop = [1.0, 1.0, 1.0]
    f_m = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    f_j_vop = [8.0, 5.0, 8.0]
    for n in range(cst.max_iteration):
        # -------------------------------------------------下面的效益函数没有考虑了VOP自身的能耗-----------------------------------------------
        # rho_m = [2 * v_number * ((lamda_m[i] / Theta_m[i]) + (Theta_m[i] ** -1 - Theta_m[i + 1] ** -1) * sum([
        #     lamda_m[j] for j in range(i + 1, v_number)])) for i in range(v_number - 1)]
        # con = [Phi_m[i] - Omega_m[i] + Pi * lamda_m[i] * v_number+1 for i in range(v_number)]
        # f_m = [con[m] / rho_m[m] if m != v_number - 1 else Theta_m[m] * con[m] / (2 * v_number * lamda_m[m]) for m in
        #        range(v_number)]
        # p_m = [(f_m[i] ** 2 / Theta_m[i] + sum(
        #     [(Theta_m[j - 1] ** -1 - Theta_m[j] ** -1) * (f_m[j - 1] ** 2) for j in range(1, v_number)])) for i in
        #        range(v_number)]
        # p_j_vop = [a * e[j] * K[j] * (sum(F[j]) + Upsilon_j[j] - Lambda_j[j]) + Pi / 2 for j in range(len(K))]
        # f_j_vop = [np.maximum(0, sum(F[j]) - p_j_vop[j] / (2 * a * e[j] * K[j])) for j in range(len(K))]
        # # f_j_vop = [sum(F[j]) - p_j_vop[j] / (2 * a * e[j] * K[j]) for j in range(len(K))]

        # #-------------------------------------------------下面的效益函数考虑了VOP向车辆购买资源的满意度-----------------------------------------------
        p_m = [0.5 * (Phi_m[m] - Omega_m[m] + Pi) for m in range(v_number)]
        f_m = [0.5 * p_m[m] * Theta_m[m] for m in range(v_number)]
        p_j_vop = [a * e[j] * K[j] * (sum(F[j]) + Upsilon_j[j] - Lambda_j[j]) + Pi / 2 for j in range(len(K))]
        f_j_vop = [np.maximum(0, sum(F[j]) - p_j_vop[j] / (2 * a * e[j] * K[j])) for j in range(len(K))]
        # -------------------------------------------------------------------------------------------------------------

        Phi_m_grad, Omega_m_grad, Pi_grad, Upsilon_j_grad, Lambda_j_grad = caculate_VopGradient(f_m, p_m, p_j_vop, F,
                                                                                                f_j_vop)
        Phi_m_new = [np.maximum(0, Phi_m[i] - cst.s_k * Phi_m_grad[i]) for i in range(v_number)]
        Omega_m_new = [np.maximum(0, Omega_m[i] - cst.s_k * Omega_m_grad[i]) for i in range(v_number)]
        Pi_new = np.maximum(0, Pi - cst.s_k * Pi_grad)
        # Pi_new = 0
        Upsilon_j_new = [np.maximum(0, Upsilon_j[j] - cst.s_k * Upsilon_j_grad[j]) for j in range(len(K))]
        Lambda_j_new = [np.maximum(0, Lambda_j[j] - cst.s_k * Lambda_j_grad[j]) for j in range(len(K))]
        # if(n!=0):
        print("第{}次迭代更新的乘子为：".format(n + 1), Phi_m_new, Omega_m_new, Pi_new, Upsilon_j_new, Lambda_j_new)
        utility_for_Vop = calculate_utility_for_Vop(f_m, p_m, p_j_vop, F)
        if np.allclose(Phi_m_new, Phi_m, atol=cst.Error_value) and np.allclose(Omega_m_new, Omega_m,
                                                                               atol=cst.Error_value) and np.allclose(
            Pi_new, Pi, atol=cst.Error_value) and np.allclose(Upsilon_j_new, Upsilon_j,
                                                              atol=cst.Error_value) and np.allclose(Lambda_j_new,
                                                                                                    Lambda_j,
                                                                                                    atol=cst.Error_value):
            break
        Phi_m = Phi_m_new
        Omega_m = Omega_m_new
        Pi = Pi_new
        Upsilon_j = Upsilon_j_new
        Lambda_j = Lambda_j_new
    p_j_vop_t, p_m_t, f_m_t = p_j_vop, p_m, f_m
    # checkConstrain(f_m, p_m, p_j_vop, F)
    return f_m, p_m, f_j_vop, p_j_vop, utility_for_Vop


p_0_t_c, p_1_t_c, p_2_t_c = [], [], []

# Initialization
p_0_min, p_1_min, p_2_min = C[0], C[1], C[2]
p_0_max, p_1_max, p_2_max = alpha[0] * beta[0] / zeta[0], alpha[1] * beta[1] / zeta[1], alpha[2] * beta[2] / zeta[2]
# p_j_min=C[j]<=p_j_max=alpha[j] * beta[j] / zeta[j]
p_0_init, p_1_init, p_2_init = 0.5 * (p_0_min + p_0_max), 0.5 * (p_1_min + p_1_max), 0.5 * (p_2_min + p_2_max)


def find_nash_equilibrium1(F, p_j_vop, f_j_vop):
    # global p_0_init, p_1_init, p_2_init
    # global p_0_t_c, p_1_t_c, p_2_t_c
    # Parameter Setup
    # Delta= 0.05

    p_0_min, p_1_min, p_2_min = C[0], C[1], C[2]
    p_0_max, p_1_max, p_2_max = alpha[0] * beta[0] / zeta[0], alpha[1] * beta[1] / zeta[1], alpha[2] * beta[2] / zeta[2]
    p_0_init, p_1_init, p_2_init = 0.5 * (p_0_min + p_0_max), 0.5 * (p_1_min + p_1_max), 0.5 * (p_2_min + p_2_max)
    Delta_0, Delta_1, Delta_2 = 1, 1, 1
    dslow, dfast = 0.6, 1.2
    p_0_t, p_1_t, p_2_t = 0, 0, 0
    n = 0
    while True:
        # Calculate utility for the cloud
        Uc = calculate_utility_for_Cloud_server(p_0_init, p_1_init, p_2_init, p_j_vop[0], f_j_vop[0])
        Uc_add_Delta = calculate_utility_for_Cloud_server(p_0_init + Delta_0, p_1_init, p_2_init, p_j_vop[0],
                                                          f_j_vop[0])
        Uc_minus_Delta = calculate_utility_for_Cloud_server(p_0_init - Delta_0, p_1_init, p_2_init, p_j_vop[0],
                                                            f_j_vop[0])

        if Uc_add_Delta >= Uc and Uc_add_Delta >= Uc_minus_Delta:
            p_0_init = min(p_0_init + Delta_0, p_0_max)
        elif Uc_minus_Delta >= Uc and Uc_minus_Delta >= Uc_add_Delta:
            p_0_init = max(p_0_init - Delta_0, p_0_min)
        else:
            p_0_init = p_0_init

        if (np.abs(p_0_init - p_0_t) <= 0.001).all():
            Delta_0 = Delta_0 * dfast
        else:
            Delta_0 = Delta_0 * dslow

        p_0_t_c.append(p_0_init)

        # Calculate utility for the M1-server
        U_m1 = calculate_utility_for_M1_server(p_1_init, p_0_init, p_2_init, p_j_vop[1], f_j_vop[1])
        U_m1_add_Delta = calculate_utility_for_M1_server(p_1_init + Delta_1, p_0_init, p_2_init, p_j_vop[1], f_j_vop[1])
        U_m1_minus_Delta = calculate_utility_for_M1_server(p_1_init - Delta_1, p_0_init, p_2_init, p_j_vop[1],
                                                           f_j_vop[1])

        if U_m1_add_Delta >= U_m1 and U_m1_add_Delta >= U_m1_minus_Delta:
            p_1_init = min(p_1_init + Delta_1, p_1_max)
        elif U_m1_minus_Delta >= U_m1 and U_m1_minus_Delta >= U_m1_add_Delta:
            p_1_init = max(p_1_init - Delta_1, p_1_min)
        else:
            p_1_init = p_1_init

        if (np.abs(p_1_init - p_1_t) <= 0.001).all():
            Delta_1 = Delta_1 * dfast
        else:
            Delta_1 = Delta_1 * dslow

        p_1_t_c.append(p_1_init)

        # Calculate utility for the M2-server
        U_m2 = calculate_utility_for_M2_server(p_2_init, p_0_init, p_1_init, p_j_vop[2], f_j_vop[2])
        U_m2_add_Delta = calculate_utility_for_M2_server(p_2_init + Delta_2, p_0_init, p_1_init, p_j_vop[2], f_j_vop[2])
        U_m2_minus_Delta = calculate_utility_for_M2_server(p_2_init - Delta_2, p_0_init, p_1_init, p_j_vop[2],
                                                           f_j_vop[2])

        if U_m2_add_Delta >= U_m2 and U_m2_add_Delta >= U_m2_minus_Delta:
            p_2_init = min(p_2_init + Delta_2, p_2_max)
        elif U_m2_minus_Delta >= U_m2 and U_m2_minus_Delta >= U_m2_add_Delta:
            p_2_init = max(p_2_init - Delta_2, p_2_min)
        else:
            p_2_init = p_2_init

        if (np.abs(p_2_init - p_2_t) <= 0.001).all():
            Delta_2 = Delta_2 * dfast
        else:
            Delta_2 = Delta_2 * dslow

        p_2_t_c.append(p_2_init)
        n = n + 1
        # print("第{}迭代定价".format(n))
        # if (p_0_t, p_1_t, p_2_t==p_0_init, p_1_init, p_2_init):
        if (p_0_t == p_0_init) and (p_1_t == p_1_init) and (p_2_t == p_2_init):
            break
        p_0_t, p_1_t, p_2_t = p_0_init, p_1_init, p_2_init
    return p_0_init, p_1_init, p_2_init
# stageII 求解算法
def find_nash_equilibrium(F, p_j_vop, f_j_vop):
    global p_0_init, p_1_init, p_2_init
    global p_0_t_c, p_1_t_c, p_2_t_c
    # Parameter Setup
    Delta= 0.05
    dslow, dfast = 0.5, 2.0

    # p_0_t, p_1_t, p_2_t = p_0_init, p_1_init, p_2_init

    # Calculate utility for the cloud
    Uc = calculate_utility_for_Cloud_server(p_0_init, p_1_init, p_2_init, p_j_vop[0], f_j_vop[0])
    Uc_add_Delta = calculate_utility_for_Cloud_server(p_0_init + Delta, p_1_init, p_2_init, p_j_vop[0], f_j_vop[0])
    Uc_minus_Delta = calculate_utility_for_Cloud_server(p_0_init - Delta, p_1_init, p_2_init, p_j_vop[0], f_j_vop[0])

    if Uc_add_Delta >= Uc and Uc_add_Delta >= Uc_minus_Delta:
        p_0_init = min(p_0_init + Delta, p_0_max)
    elif Uc_minus_Delta >= Uc and Uc_minus_Delta >= Uc_add_Delta:
        p_0_init = max(p_0_init - Delta, p_0_min)
    else:
        p_0_init = p_0_init

    p_0_t_c.append(p_0_init)

    # Calculate utility for the M1-server
    U_m1 = calculate_utility_for_M1_server(p_1_init, p_0_init, p_2_init, p_j_vop[1], f_j_vop[1])
    U_m1_add_Delta = calculate_utility_for_M1_server(p_1_init + Delta, p_0_init, p_2_init, p_j_vop[1], f_j_vop[1])
    U_m1_minus_Delta = calculate_utility_for_M1_server(p_1_init - Delta, p_0_init, p_2_init, p_j_vop[1], f_j_vop[1])

    if U_m1_add_Delta >= U_m1 and U_m1_add_Delta >= U_m1_minus_Delta:
        p_1_init = min(p_1_init + Delta, p_1_max)
    elif U_m1_minus_Delta >= U_m1 and U_m1_minus_Delta >= U_m1_add_Delta:
        p_1_init = max(p_1_init - Delta, p_1_min)
    else:
        p_1_init = p_1_init

    p_1_t_c.append(p_1_init)

    # Calculate utility for the M2-server
    U_m2 = calculate_utility_for_M2_server(p_2_init, p_0_init, p_1_init, p_j_vop[2], f_j_vop[2])
    U_m2_add_Delta = calculate_utility_for_M2_server(p_2_init + Delta, p_0_init, p_1_init, p_j_vop[2], f_j_vop[2])
    U_m2_minus_Delta = calculate_utility_for_M2_server(p_2_init - Delta, p_0_init, p_1_init, p_j_vop[2], f_j_vop[2])

    if U_m2_add_Delta >= U_m2 and U_m2_add_Delta >= U_m2_minus_Delta:
        p_2_init = min(p_2_init + Delta, p_2_max)
    elif U_m2_minus_Delta >= U_m2 and U_m2_minus_Delta >= U_m2_add_Delta:
        p_2_init = max(p_2_init - Delta, p_2_min)
    else:
        p_2_init = p_2_init

    p_2_t_c.append(p_2_init)

    return p_0_init, p_1_init, p_2_init


# stageIII 求解算法
def optimal_Stage3strategy_KKT(bi, P_0, P_1, P_2):
    # def optimal_Stage3strategy_KKT(bi):
    # global alpha, beta, zeta
    P = np.array([P_0, P_1, P_2])
    # P = [P_0, P_1, P_2]
    ################################################################### Case 1
    if np.allclose(P, alpha * beta / zeta):
        # print("Casse 1")
        f_i0, f_i1, f_i2 = 0, 0, 0
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 2-1
    f_i0 = alpha[0] / (P[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (P[1] * zeta[1]) - beta[1] ** -1
    f_i2 = alpha[2] / (P[2] * zeta[2]) - beta[2] ** -1
    if f_i0 > 0 and f_i1 > 0 and f_i2 > 0 and (bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        # print("Casse 2-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 2-2
    A = np.sum(alpha)
    B = np.sum(P / beta)
    lamda = (A / (B + bi)) - zeta[0]
    f_i0 = alpha[0] * (B + bi) / (A * P[0]) - beta[0] ** -1
    f_i1 = alpha[1] * (B + bi) / (A * P[1]) - beta[1] ** -1
    f_i2 = alpha[2] * (B + bi) / (A * P[2]) - beta[2] ** -1

    f_i0 = alpha[0] / ((lamda + zeta[0]) * P[0]) - beta[0] ** -1
    # f_i1 = alpha[1] * (B + bi) / (A * P[1]) - beta[1] ** -1
    # f_i2 = alpha[2] * (B + bi) / (A * P[2]) - beta[2] ** -1
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and f_i0 > 0 and f_i1 > 0 and f_i2 > 0:
        # print("Casse 2-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 3-1
    f_i0 = 0
    f_i1 = alpha[1] / (zeta[1] * P[1]) - beta[1] ** -1
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and f_i1 > 0 and f_i2 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        # print("Casse 3-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 3-2
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 0  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i
    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B = np.sum(P[mask] / beta[mask])
    # B = (P[1] / beta[1])+P[2] / beta[2]
    lamda = (A / (B + bi)) - zeta[i]
    mu_i0 = P[i] * A / (B + bi) - alpha[i] * beta[i]
    f_i0 = 0
    f_i1 = alpha[1] * (B + bi) / (A * P[1]) - beta[1] ** -1
    f_i2 = alpha[2] * (B + bi) / (A * P[2]) - beta[2] ** -1
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i1 > 0 and f_i2 > 0:
        # print("Casse 3-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 4-1
    f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and f_i0 > 0 and f_i2 > 0 and (
            bi - sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                      range(nuser)])) >= 0:
        # print("Casse 4-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################## Case 4-2
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 1  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i

    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B = np.sum(P[mask] / beta[mask])
    # B = (P[1] / beta[1])+P[2] / beta[2]
    lamda = (A / (B + bi)) - zeta[i]
    mu_i1 = P[i] * A / (B + bi) - alpha[i] * beta[i]
    f_i1 = 0
    f_i0 = alpha[0] * (B + bi) / (A * P[0]) - beta[0] ** -1
    f_i2 = alpha[2] * (B + bi) / (A * P[2]) - beta[2] ** -1
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i1 >= 0 and f_i0 > 0 and f_i2 > 0:
        # print("Casse 4-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 5-1
    f_i2 = 0
    f_i0 = alpha[0] / (P[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (P[1] * zeta[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and f_i1 > 0 and f_i2 > 0 and (
            bi - sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                      range(nuser)])) >= 0:
        # print("Casse 5-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 5-2
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 2  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i

    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B = np.sum(P[mask] / beta[mask])
    lamda = (A / (B + bi)) - zeta[i]
    mu_i0 = P[i] * A / (B + bi) - alpha[i] * beta[i]
    f_i2 = 0
    f_i1 = alpha[1] * (B + bi) / (A * P[1]) - beta[1] ** -1
    f_i0 = alpha[0] * (B + bi) / (A * P[0]) - beta[0] ** -1
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i0 > 0 and f_i1 > 0:
        # print("Casse 5-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 6-1
    f_i0 = f_i1 = 0
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and P[1] == alpha[1] * beta[1] / zeta[1] and f_i2 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        # print("Casse 6-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 6-2
    mask = 2
    f_i0 = f_i1 = 0
    f_i2 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i0 = lamda * P[0] - alpha[0] * beta[0]
    mu_i1 = lamda * P[1] - alpha[1] * beta[1]
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i1 >= 0:
        # print("Casse 6-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 7-1
    f_i0 = f_i2 = 0
    f_i1 = alpha[1] / (zeta[1] * P[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i1 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        # print("Casse 7-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 7-2
    mask = 1
    f_i0 = f_i2 = 0
    f_i1 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i0 = lamda * P[0] - alpha[0] * beta[0]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i2 >= 0:
        # print("Casse 7-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 8-1
    f_i2 = f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i0 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        # print("Casse 8-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 8-2
    mask = 0
    f_i2 = f_i1 = 0
    f_i0 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i1 = lamda * P[1] - alpha[1] * beta[1]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    # b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i2 >= 0 and mu_i1 >= 0:
        # print("Casse 8-2")
        return bi, f_i0, f_i1, f_i2

    return bi, 0, 0, 0


if __name__ == '__main__':
    cst.UserDevice.read(nuser)
    n_vechicle = 8
    U_C_t_v, U_M1_t_v, U_M2_t_v = [], [], []
    utility_for_user_device_t_v, utility_for_Vop_t_v = [], []
    utility_for_user_device_t, utility_for_Vop_t = [0 for i in range(nuser)], 0
    average_utility_for_user_v = []
    P_0_v, P_1_v, P_2_v, p_j_vop_v = [], [], [], []
    F_0_v, F_1_v, F_2_v, F_j_vop_v = [], [], [], []
    f_m_v, p_m_v = [], []
    for vumber in range(n_vechicle, n_vechicle+1):
        # p_0_init, p_1_init, p_2_init= 0.6, 0.3, 0.3
        P_0, P_1, P_2 = 0.6, 0.3, 0.3
        f_m, p_m = [], []  # 合同（f_m,p_m）
        f_j_vop, p_j_vop = [0.6, 0.3, 0.3], [0.6, 0.3, 0.3]  # CEA的资源购买决策、vop的定价
        U_C, U_M1, U_M2 = 0, 0, 0
        U_C_t, U_M1_t, U_M2_t = 0, 0, 0
        P_0_t, P_1_t, P_2_t = 0.6, 0.3, 0.3
        n = 1
        v_number = vumber
        cst.LM.read(v_number)
        cst.Vechicle.read(v_number)
        create()
        print(
            "--------------------------------------------------------------------------nuser={}--------------------------------------------------------------------------：".format(
                nuser))
        while True:
            print(
                "--------------------------------------------------------------------------第{}次博弈,nuser={}--------------------------------------------------------------------------：".format(
                    n, nuser))
            # Algorithm 1
            F_i0, F_i1, F_i2 = find_Optial_mulitUser(P_0, P_1, P_2)
            print("stageIII的购买决策F_i0, F_i1, F_i2分别为:", F_i0, F_i1, F_i2)
            F = [F_i0, F_i1, F_i2]
            # Algorithm 2
            P_0, P_1, P_2 = find_nash_equilibrium(F, p_j_vop, f_j_vop)  # 这里加了一个p_j_vop, f_j_vop

            # Algorithm 3
            f_m, p_m, f_j_vop, p_j_vop, utility_for_Vop = LagrangeDualStageIforVop(F)

            if ([0 <= p_j_vop[j] <= sum(F[j]) * 2 * a * e[j] * K[j] for j in range(len(K))]) is False:
                print("不满足条件")

            print("多购买了{}的资源".format(v_number * sum([lamda_m[i] * f_m[i] for i in range(v_number)]) - sum(f_j_vop)))
            print("stageI阶段合同为f_m, p_m：", f_m, p_m)
            print("stageI阶段Vop对CEA的资源定价p_j_vop为", p_j_vop)
            print("stageII阶段CEA的价格P_0, P_1, P_2分别为：", P_0, P_1, P_2)
            print("stageII阶段CEA的资源购买决策f_j_vop：", f_j_vop)
            utility_for_user_device = calculate_utility_for_user_device(F_i0, F_i1, F_i2, P_0, P_1, P_2)
            U_C = calculate_utility_for_Cloud_server(P_0, P_1, P_2, p_j_vop[0], f_j_vop[0])
            U_M1 = calculate_utility_for_M1_server(P_1, P_0, P_2, p_j_vop[1], f_j_vop[1])
            U_M2 = calculate_utility_for_M2_server(P_2, P_0, P_1, p_j_vop[2], f_j_vop[2])
            utility_for_Vop = calculate_utility_for_Vop(f_m, p_m, p_j_vop, F)

            print("------------------------------------------")
            print("user的效益函数为：", utility_for_user_device)
            print("U_C效益函数为：", U_C)
            print("U_M1效益函数为：", U_M1)
            print("U_M2效益函数为：", U_M2)
            print("------------------------------------------")
            print("stageI阶段Vop的效益函数为", utility_for_Vop)
            print("------------------------------------------")
            print("整体社会效益为", sum(utility_for_user_device) + U_C + U_M1 + U_M2 + utility_for_Vop)
            print("------------------------------------------")
            # if (np.abs(P_0_t - P_0) <= cst.epsilon).all() and (np.abs(P_1_t - P_1) <= cst.epsilon).all() and (
            #         np.abs(P_2_t - P_2) <= cst.epsilon).all():
            #     break
            P_0_t, P_1_t, P_2_t = P_0, P_1, P_2
            if (np.abs(U_C - U_C_t) <= cst.Error_value).all() and (np.abs(U_M1 - U_M1_t) <= cst.Error_value).all() and (
                    np.abs(U_M2 - U_M2_t) <= cst.Error_value).all() and all(
                diff < 1 for diff in [np.abs(a - b) for a, b in
                                      zip(utility_for_user_device_t,
                                          utility_for_user_device)]) and (
                    np.abs(utility_for_Vop - utility_for_Vop_t) <= cst.Error_value).all():
                break
            U_C_t = U_C
            U_M1_t = U_M1
            U_M2_t = U_M2
            utility_for_user_device_t = utility_for_user_device
            utility_for_Vop_t = utility_for_Vop
            n += 1

        print("------------------------------------------user={}".format(nuser))
        print("已达到纳什均衡")
        U_C_t_v.append(U_C)
        U_M1_t_v.append(U_M1)
        U_M2_t_v.append(U_M2)
        utility_for_user_device_t_v.append(utility_for_user_device)
        utility_for_Vop_t_v.append(utility_for_Vop)
        average_utility_for_user_v.append(np.average(utility_for_user_device))
        P_0_v.append(P_0), P_1_v.append(P_1), P_2_v.append(P_2), p_j_vop_v.append(p_j_vop)
        F_0_v.append(sum(F_i0)), F_1_v.append(sum(F_i1)), F_2_v.append(sum(F_i2)), F_j_vop_v.append(f_j_vop)
        f_m_v.append(f_m), p_m_v.append(p_m)
    print("stageI阶段合同为f_m, p_m：", f_m_v, p_m_v)
    print("P_0_v,P_1_v,P_2_v,p_j_vop_v=", P_0_v, ',', P_1_v, ',', P_2_v, ',', p_j_vop_v)
    print("F_0_v,F_1_v,F_2_v,f_j_vop_v=", F_0_v, ',', F_1_v, ',', F_2_v, ',', F_j_vop_v)
    # print("--------------------------f_m, p_m, p_j_vop-------------------", f_m, ',', p_m, ',', p_j_vop)
    print("U_user_v,U_C_t_v, U_M1_t_v, U_M2_t_v, U_vop_v=",
          average_utility_for_user_v, ',', U_C_t_v, ',', U_M1_t_v, ',',
          U_M2_t_v, ',', utility_for_Vop_t_v)
    print("用户平均效益值", average_utility_for_user_v)
    utilityTorTotalVechicle = sum([p_m[m] * f_m[m] - (f_m[m] ** 2 / Theta_m[m]) for m in range(v_number)])
    print("车辆整体效益值", utilityTorTotalVechicle)
    print("整体社会效益为", sum(utility_for_user_device) + U_C + U_M1 + U_M2 + utility_for_Vop + utilityTorTotalVechicle)

    # checkConstrain(f_m, p_m, p_j_vop, F)
