import game.Config.constants as cst

import numpy as np

nuser = cst.n_number
bg = cst.bg

# 拉格朗日乘子
phi_0, phi_1, phi_2, phi_3, phi_4 = 8.2, 2.0, 9.8, 4.9, 1.9
Mu_0, Mu_1, Mu_2, Mu_3, Mu_4 = 5.4, 1.4, 1.5, 4.6, 6.2
theta_0, theta_1, theta_2, theta_3, theta_4 = 4.9, 9.2, 5.0, 7.3, 9.5
# 权重参数
alpha, beta, zeta = np.array([2.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([1.65, 1.65, 1.65])
varphi, e, a = 8.2, 2.0, 9.8
C = [0.4, 0.2, 0.2]  # 三台服务器成本
K = [2, 1, 1]  # 服务器功率
f_1_max, f_2_max = 5, 7  # M1、M2最大能提供的计算资源

# stage 3决策变量
P_rsu = 0.2


# stage 2决策变量
# P_0, P_1, P_2 = 0.6, 0.3, 0.3
# f_0_rsu, f_1_rsu, f_2_rsu= 0.6, 0.3, 0.3
# f_0, f_1, f_2 = 0.6, 0.3, 0.3  # 服务器能够提供的计算资源
# stage 1决策变量
# F_i0, F_i1, F_i2 = np.array(np.zeros((1, nuser))), np.array(np.zeros((1, nuser))), np.array(np.zeros((1, nuser)))
# F_i0, F_i1, F_i2 = [], [], []


def optimal_Stage3strategy_KKT(bi, P_0, P_1, P_2):
    global alpha, beta, zeta
    # global P_0, P_1, P_2
    P = np.array([P_0, P_1, P_2])
    ################################################################### Case 1
    if np.allclose(P, alpha * beta / zeta):
        print("Case 1")
        f_i0, f_i1, f_i2 = 0, 0, 0
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 2-1
    f_i0 = alpha[0] / (P[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (P[1] * zeta[1]) - beta[1] ** -1
    f_i2 = alpha[2] / (P[2] * zeta[2]) - beta[2] ** -1
    if f_i0 > 0 and f_i1 > 0 and f_i2 > 0 and (bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 2-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 2-2
    A = np.sum(alpha)
    B = np.sum(P / beta)
    lamda = (A / (B + bi)) - zeta[0]
    f_i0 = alpha[0] * (B + bi) / (A * P[0]) - beta[0] ** -1
    f_i1 = alpha[1] * (B + bi) / (A * P[1]) - beta[1] ** -1
    f_i2 = alpha[2] * (B + bi) / (A * P[2]) - beta[2] ** -1
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda  > 0 and f_i0 > 0 and f_i1 > 0 and f_i2 > 0:
        print("Case 2-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 3-1
    f_i0 = 0
    f_i1 = alpha[1] / (zeta[1] * P[1]) - beta[1] ** -1
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and f_i1 > 0 and f_i2 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 3-1")
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
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i1 > 0 and f_i2 > 0:
        print("Case 3-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 4-1
    f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and f_i0 > 0 and f_i2 > 0 and (
            bi - sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                      range(nuser)])) >= 0:
        print("Case 4-1")
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
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i1 >= 0 and f_i0 > 0 and f_i2 > 0:
        print("Case 4-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 5-1
    f_i2 = 0
    f_i0 = alpha[0] / (P[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (P[1] * zeta[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and f_i1 > 0 and f_i2 > 0 and (
            bi - sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                      range(nuser)])) >= 0:
        print("Case 5-1")
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
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i0 > 0 and f_i1 > 0:
        print("Case 5-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 6-1
    f_i0 = f_i1 = 0
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and P[1] == alpha[1] * beta[1] / zeta[1] and f_i2 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 6-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 6-2
    mask = 2
    f_i0 = f_i1 = 0
    f_i2 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i0 = lamda * P[0] - alpha[0] * beta[0]
    mu_i1 = lamda * P[1] - alpha[1] * beta[1]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i1 >= 0:
        print("Case 6-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 7-1
    f_i0 = f_i2 = 0
    f_i1 = alpha[1] / (zeta[1] * P[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i1 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 7-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 7-2
    mask = 1
    f_i0 = f_i2 = 0
    f_i1 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i0 = lamda * P[0] - alpha[0] * beta[0]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i2 >= 0:
        print("Case 7-2")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 8-1
    f_i2 = f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i0 > 0 and (
            bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 8-1")
        return bi, f_i0, f_i1, f_i2

    ################################################################### Case 8-2
    mask = 0
    f_i2 = f_i1 = 0
    f_i0 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i1 = lamda * P[1] - alpha[1] * beta[1]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i2 >= 0 and mu_i1 >= 0:
        print("Case 8-2")
        return bi, f_i0, f_i1, f_i2

    return bi, 0, 0, 0


# 计算效益函数值
def caculateUilityofCloudAndMec():
    f = [f_0, f_1, f_2]
    P = [P_0, P_1, P_2]
    F = [np.sum(F_i0), np.sum(F_i1), np.sum(F_i2)]
    # 计算 能耗成本E
    E = e * K * (f ** 2)
    # 计算整体表达式
    U = varphi * np.log(1 + P - C) * f - a * E - P_rsu * (F - f)
    return U


# 计算cloud梯度
def caculate_CloudGradient(F_i0, f_0, P_0):
    phi_0_grad = P_0 - C[0]
    phi_1_grad = np.sum(F_i0) - f_0
    phi_2_grad = f_0
    phi_4_grad = alpha[0] * beta[0] - zeta[0] * P_0
    return phi_0_grad, phi_1_grad, phi_2_grad, phi_4_grad


# 计算M1梯度
def caculate_MEC1Gradient(F_i1, f_1, P_1):
    phi_0_grad = P_1 - C[1]
    phi_1_grad = np.sum(F_i1) - f_1
    phi_2_grad = f_1
    phi_3_grad = f_1_max - f_1
    phi_4_grad = alpha[1] * beta[1] - zeta[1] * P_1
    return phi_0_grad, phi_1_grad, phi_2_grad, phi_3_grad, phi_4_grad


# 计算M2梯度
def caculate_MEC2Gradient(F_i2, f_2, P_2):
    phi_0_grad = P_2 - C[2]
    phi_1_grad = np.sum(F_i2) - f_2
    phi_2_grad = f_2
    phi_3_grad = f_2_max - f_2
    phi_4_grad = alpha[2] * beta[2] - zeta[2] * P_2
    return phi_0_grad, phi_1_grad, phi_2_grad, phi_3_grad, phi_4_grad


# 核心代码：拉格朗日交替更新拉格朗日乘子
def LagrangeDual(F_i0, F_i1, F_i2, f_0, f_1, f_2, P_0, P_1, P_2):
    # cloud 迭代
    # global f_0, P_0  # f_0 可提供的计算资源、P_0出售的价格
    # global varphi, c, e, k, a, beta, zeta
    global phi_0, phi_1, phi_2, phi_3, phi_4
    for n in range(cst.max_iteration):
        if phi_4 * zeta[0] - phi_0 == 0:
            # P_0_new=alpha[0]*beta[0]/zeta[0]
            P_0_new = C[0]
        else:
            P_0_new = np.maximum(C[0], varphi * f_0 / (phi_4 * zeta[0] - phi_0) + C[0] - 1)
        f_0_new = np.maximum(0, (varphi * np.log(1 + P_0 - C[0]) + P_rsu - phi_1 + phi_2) / (2 * a * e * K[0]))
        f_0, P_0 = f_0_new, P_0_new
        phi_0_grad, phi_1_grad, phi_2_grad, phi_4_grad = caculate_CloudGradient(F_i0, f_0, P_0)
        phi_0_new = np.maximum(0, phi_0 + cst.s_k * phi_0_grad)
        phi_1_new = np.maximum(0, phi_1 + cst.s_k * phi_1_grad)
        phi_2_new = np.maximum(0, phi_2 + cst.s_k * phi_2_grad)
        phi_4_new = np.maximum(0, phi_4 + cst.s_k * phi_4_grad)
        if (np.abs(phi_0_new - phi_0) <= cst.Error_value).all() and (
                np.abs(phi_1_new - phi_1) <= cst.Error_value).all() and (
                np.abs(phi_2_new - phi_2) <= cst.Error_value).all() and (
                np.abs(phi_4_new - phi_4) <= cst.Error_value).all():
            break
            # return P_0_new, f_0_new
        phi_0 = phi_0_new
        phi_1 = phi_1_new
        phi_2 = phi_2_new
        phi_4 = phi_4_new

    # M1 迭代
    # global f_1, P_1  # f_0 可提供的计算资源、P_0出售的价格
    global Mu_0, Mu_1, Mu_2, Mu_3, Mu_4
    for n in range(cst.max_iteration):
        if Mu_4 * zeta[1] - Mu_0 == 0:
            # P_1_new=alpha[1]*beta[1]/zeta[1]
            P_1_new = C[1]
        else:
            P_1_new = varphi * f_1 / (phi_4 * zeta[1] - Mu_0) + C[1] - 1

        f_1_new = (varphi * np.log(1 + P_1 - C[1]) + P_rsu - Mu_1 + Mu_2 - Mu_3) / (2 * a * e * K[1])
        f_1, P_1 = f_1_new, P_1_new
        Mu_0_grad, Mu_1_grad, Mu_2_grad, Mu_3_grad, Mu_4_grad = caculate_MEC1Gradient(F_i1, f_1, P_1)
        Mu_0_new = np.maximum(0, Mu_0 + cst.s_k * Mu_0_grad)
        Mu_1_new = np.maximum(0, Mu_1 + cst.s_k * Mu_1_grad)
        Mu_2_new = np.maximum(0, Mu_2 + cst.s_k * Mu_2_grad)
        Mu_3_new = np.maximum(0, Mu_3 + cst.s_k * Mu_3_grad)
        Mu_4_new = np.maximum(0, Mu_4 + cst.s_k * Mu_4_grad)
        if (np.abs(Mu_0_new - Mu_0) <= cst.Error_value).all() and (
                np.abs(Mu_1_new - Mu_1) <= cst.Error_value).all() and (
                np.abs(Mu_2_new - Mu_2) <= cst.Error_value).all() and (
                np.abs(Mu_3_new - Mu_3) <= cst.Error_value).all() and (
                np.abs(Mu_4_new - Mu_4) <= cst.Error_value).all():
            break
            # return P_0_new, f_0_new
        Mu_0 = Mu_0_new
        Mu_1 = Mu_1_new
        Mu_2 = Mu_2_new
        Mu_3 = Mu_3_new
        Mu_4 = Mu_4_new

    # M2 迭代
    # global f_2, P_2  # f_0 可提供的计算资源、P_0出售的价格
    global theta_0, theta_1, theta_2, theta_3, theta_4
    for n in range(cst.max_iteration):
        if theta_4 * zeta[1] - theta_0 == 0:
            P_2_new = alpha[2] * beta[2] / zeta[2]
            # P_2_new = C[2]
        else:
            P_2_new = varphi * f_2 / (theta_4 * zeta[2] - theta_0) + C[2] - 1
        f_2_new = (varphi * np.log(1 + P_2 - C[2]) + P_rsu - theta_1 + theta_2 - theta_3) / (2 * a * e * K[1])
        f_2, P_2 = f_2_new, P_2_new
        theta_0_grad, theta_1_grad, theta_2_grad, theta_3_grad, theta_4_grad = caculate_MEC2Gradient(F_i2, f_2, P_2)
        theta_0_new = np.maximum(0, theta_0 + cst.s_k * theta_0_grad)
        theta_1_new = np.maximum(0, theta_1 + cst.s_k * theta_1_grad)
        theta_2_new = np.maximum(0, theta_2 + cst.s_k * theta_2_grad)
        theta_3_new = np.maximum(0, theta_3 + cst.s_k * theta_3_grad)
        theta_4_new = np.maximum(0, theta_4 + cst.s_k * theta_4_grad)
        if (np.abs(theta_0_new - theta_0) <= cst.Error_value).all() and (
                np.abs(theta_1_new - theta_1) <= cst.Error_value).all() and (
                np.abs(theta_2_new - theta_2) <= cst.Error_value).all() and (
                np.abs(theta_3_new - theta_3) <= cst.Error_value).all() and (
                np.abs(theta_4_new - theta_4) <= cst.Error_value).all():
            break
            # return P_0_new, f_0_new
        theta_0 = theta_0_new
        theta_1 = theta_1_new
        theta_2 = theta_2_new
        theta_3 = theta_3_new
        theta_4 = theta_4_new
    return P_0, P_1, P_2, f_0, f_1, f_2


if __name__ == '__main__':

    P_0, P_1, P_2 = 0.6, 0.3, 0.3
    f_0, f_1, f_2 = 0.6, 0.3, 0.3  # 服务器能够提供的计算资源
    while 1 == 1:
        cst.UserDevice.read(nuser)
        # cst.Phi.read(nuser)
        F_i0, F_i1, F_i2 = [], [], []
        for i in range(nuser):
            result = optimal_Stage3strategy_KKT(bg[i], P_0, P_1, P_2)
            F_i0 = np.append(F_i0, result[1])
            F_i1 = np.append(F_i1, result[2])
            F_i2 = np.append(F_i2, result[3])
        print(F_i0 + F_i1 + F_i2)
        P_0, P_1, P_2, f_0, f_1, f_2 = LagrangeDual(F_i0, F_i1, F_i2, f_0, f_1, f_2, P_0, P_1, P_2)
