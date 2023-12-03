import game.Config.constants as cst

import numpy as np

nuser = cst.n_number
bg = cst.bg
f_i0 = np.array(np.zeros((1, nuser)))
f_i1 = np.array(np.zeros((1, nuser)))
f_i2 = np.array(np.zeros((1, nuser)))

def optimal_Stage3strategy_KKT(P_0, P_1, P_2, alpha, beta, zeta,bi):
    ################################################################### Case 1
    P = np.array([P_0, P_1, P_2])
    if np.allclose(P, alpha * beta / zeta):
        print("Case 1")
        return bi,0, 0, 0

    ################################################################### Case 2-1
    P = np.array([P_0, P_1, P_2])
    f_i0 = alpha[0] / (P[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (P[1] * zeta[1]) - beta[1] ** -1
    f_i2 = alpha[2] / (P[2] * zeta[2]) - beta[2] ** -1
    if f_i0 > 0 and f_i1 > 0 and f_i2 > 0 and (bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 2-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 2-2
    P = np.array([P_0, P_1, P_2])
    A=np.sum(alpha)
    B=np.sum(P/beta)
    lamda = (A / (B + bi))-zeta[0]
    f_i0= alpha[0]*(B+bi)/(A*P[0])-beta[0]**-1
    f_i1= alpha[1]*(B+bi)/(A*P[1])-beta[1]**-1
    f_i2= alpha[2]*(B+bi)/(A*P[2])-beta[2]**-1
    b= bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])
    if lamda and f_i0 > 0 and f_i1 > 0 and f_i2 > 0:
        print("Case 2-2")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 3-1
    P = np.array([P_0, P_1, P_2])
    f_i0= 0
    f_i1= alpha[1]/(zeta[1]*P[1])-beta[1]**-1
    f_i2= alpha[2]/(zeta[2]*P[2])-beta[2]**-1
    if P[0] == alpha[0]*beta[0]/zeta[0] and f_i1 > 0 and f_i2 > 0 and (bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 3-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 3-2
    P = np.array([P_0, P_1, P_2])
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 0  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i
    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B= np.sum(P[mask] / beta[mask])
    # B = (P[1] / beta[1])+P[2] / beta[2]
    lamda = (A / (B + bi))-zeta[i]
    mu_i0=P[i]*A/(B + bi)-alpha[i]*beta[i]
    f_i0 = 0
    f_i1 = alpha[1] * (B + bi) / (A * P[1])-beta[1]**-1
    f_i2 = alpha[2] * (B + bi) / (A * P[2])-beta[2]**-1
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i1 > 0 and f_i2 > 0:
        print("Case 3-2")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 4-1
    P = np.array([P_0, P_1, P_2])
    f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    f_i2 = alpha[2] / (zeta[2] * P[2]) - beta[2] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and f_i0 > 0 and f_i2 > 0 and (bi -sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                  range(nuser)])) >= 0:
        print("Case 4-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################## Case 4-2
    P = np.array([P_0, P_1, P_2])
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 1  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i

    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B= np.sum(P[mask] / beta[mask])
    # B = (P[1] / beta[1])+P[2] / beta[2]
    lamda = (A / (B + bi))-zeta[i]
    mu_i1 =P[i] * A / (B + bi)  - alpha[i] * beta[i]
    f_i1 = 0
    f_i0 = alpha[0] * (B + bi) / (A * P[0])-beta[0]**-1
    f_i2 = alpha[2] * (B + bi) / (A * P[2])-beta[2]**-1
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i1 >= 0 and f_i0 > 0 and f_i2 > 0:
        print("Case 4-2")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 5-1
    P = np.array([P_0, P_1, P_2])
    f_i2 = 0
    f_i0 = alpha[0] / (p[0] * zeta[0]) - beta[0] ** -1
    f_i1 = alpha[1] / (p[1] * zeta[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and f_i1 > 0 and f_i2 > 0 and (
            bi - sum([f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2] for i in
                  range(nuser)])) >= 0:
        print("Case 5-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 5-2
    P = np.array([P_0, P_1, P_2])
    # 假设要对数组元素进行求和，但不包括下标为 i 的元素
    i = 2  # 这里以 0 为例，你可以根据实际情况更改 i 的值
    # 使用逻辑运算符创建一个布尔掩码
    mask = np.arange(3) != i

    # 对数组进行求和，但不包括下标为 i 的元素
    A = np.sum(alpha[mask])
    B= np.sum(P[mask] / beta[mask])
    lamda = (A / (B + bi))-  zeta[i]
    mu_i0 =P[i] * A / (B + bi)  - alpha[i] * beta[i]
    f_i2 = 0
    f_i1 = alpha[1] * (B + bi) / (A * P[1])-beta[1]**-1
    f_i0 = alpha[0] * (B + bi) / (A * P[0])-beta[0]**-1
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and f_i0 > 0 and f_i1 > 0:
        print("Case 5-2")
        return bi,f_i0, f_i1, f_i2


    ################################################################### Case 6-1
    P = np.array([P_0, P_1, P_2])
    f_i0= f_i1= 0
    f_i2= alpha[2]/(zeta[2]*P[2])-beta[2]**-1
    if P[0] == alpha[0]*beta[0]/zeta[0] and P[1] == alpha[1]*beta[1]/zeta[1] and f_i2 > 0 and (bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 6-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 6-2
    P = np.array([P_0, P_1, P_2])
    mask = 2
    f_i0= f_i1= 0
    f_i2 = bi/P[mask]
    lamda = alpha[mask]*beta[mask]/(P[mask]+beta[mask]*bi) -zeta[mask]
    mu_i0=lamda*P[0]-alpha[0]*beta[0]
    mu_i1=lamda*P[1]-alpha[1]*beta[1]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i1 >= 0:
        print("Case 6-2")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 7-1
    P = np.array([P_0, P_1, P_2])
    f_i0 = f_i2 = 0
    f_i1 = alpha[1] / (zeta[1] * P[1]) - beta[1] ** -1
    if P[0] == alpha[0] * beta[0] / zeta[0] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i1 > 0 and (bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 7-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 7-2
    P = np.array([P_0, P_1, P_2])
    mask = 1
    f_i0 = f_i2 = 0
    f_i1 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i0 = lamda * P[0] - alpha[0] * beta[0]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i0 >= 0 and mu_i2 >= 0:
        print("Case 7-2")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 8-1
    P = np.array([P_0, P_1, P_2])
    f_i2 = f_i1 = 0
    f_i0 = alpha[0] / (zeta[0] * P[0]) - beta[0] ** -1
    if P[1] == alpha[1] * beta[1] / zeta[1] and P[2] == alpha[2] * beta[2] / zeta[2] and f_i0 > 0 and (bi - (f_i0 * P[0]+ f_i1 * P[1] + f_i2 * P[2])) >= 0:
        print("Case 8-1")
        return bi,f_i0, f_i1, f_i2

    ################################################################### Case 8-2
    P = np.array([P_0, P_1, P_2])
    mask = 0
    f_i2 = f_i1 = 0
    f_i0 = bi / P[mask]
    lamda = alpha[mask] * beta[mask] / (P[mask] + beta[mask] * bi) - zeta[mask]
    mu_i1 = lamda * P[1] - alpha[1] * beta[1]
    mu_i2 = lamda * P[2] - alpha[2] * beta[2]
    b = bi - (f_i0 * P[0] + f_i1 * P[1] + f_i2 * P[2])
    if lamda > 0 and mu_i2 >= 0 and mu_i1 >= 0:
        print("Case 8-2")
        return bi,f_i0, f_i1, f_i2


    return bi,0,0,0


# 计算效益函数值
def caculateUilityofCloudAndMec(p, c, f, a, p_Rsu, f_Rsu, e, K,phi):
    # 定义变量
    # 计算 能耗成本E
    E = e * K * f ** 2
    # 计算整体表达式
    U_c = phi * np.log(1 + p - c) * f - a * E - p_Rsu * f_Rsu
    return U_c

# 计算梯度
def caculate_gradient(f_c):
    phi_2_grad=f_c
    return 0


# 计算Cloud效益函数值
def caculateUilityofCloud(p, c, f, a, p_Rsu, e, K,phi,f_ic):
    # 定义变量
    # 计算 能耗成本E
    E = e * K * f ** 2
    # 计算整体表达式
    U_c = phi * np.log(1 + p - c) * f - a * E - p_Rsu * (np.sum(f_ic)-f)
    return U_c

# 计算梯度
def caculate_gradient(f_c,f_ic,a,beta,zeta,p_c):
    phi_1_grad=np.sum(f_ic)-f_c
    phi_2_grad=f_c
    phi_4_grad=a*beta-zeta*p_c
    return phi_1_grad,phi_2_grad,phi_4_grad


# 核心代码：拉格朗日交替更新拉格朗日乘子
def LagrangeDual(fi_0,fi_1,fi_2):
    f_c, f_ic, a, beta, zeta, p_c=0
    phi_1_grad,phi_2_grad,phi_4_grad=caculate_gradient(f_c, f_ic, a, beta, zeta, p_c)
    return 0




H = 2
R = 3
N = 4
a = 0.5
b = 0.2
bi = 0.8
p = [10, 5, 4]
pt = 5


# P_0,P_1,P_2=1,0.5,0.4
P_0,P_1,P_2=0.6,0.3,0.3
alpha, beta, zeta=np.array([0.7,1.4,1.4]),np.array([0.5,0.5,0.5]),np.array([1,1.2,1.3])

# P_0,P_1,P_2=23,2,14
# alpha, beta, zeta=np.array([7,14,12]),np.array([7.2,8.2,545.2]),np.array([71,81,5])
if __name__ == '__main__':
    cst.UserDevice.read(nuser)
    ls=[]
    for i in range(nuser):
        result = optimal_Stage3strategy_KKT(P_0, P_1, P_2, alpha, beta, zeta, bg[i])
        ls.append(result)
        print(result)
    print(ls)
