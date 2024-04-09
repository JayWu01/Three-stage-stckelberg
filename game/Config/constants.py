import numpy as np
import csv
import os
import random

# # 常量
# n_number = 100
# v_number = 30

n_number = 10
v_number = 10
ecsp_number=3
bg = []
s_k =0.04# (0.8-1.4)
# s_k =0.1# (0.8-1.4)
# s_k = 1.5 # (0.8-1.4)

Error_value = 0.00001
# Error_value = 0.000000000000001
max_iteration = 2000
epsilon=0.01   # 博弈停止的精度阈值
error_price=0.001
# 变量 按照一定规则生成
# 初始化乘子 1*n_number的矩阵  （n_number代表用户数量）
lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = [], [], [], [], [], [], [], [], []
Omega_m, Phi_m = [], []
Rho_m=[]  # f_m递增约束
canshu = [1, 5]

file_path = 'D:/江西理工大学/边缘计算/Three-stage-stckelberg/game/Data/'
# file_path = '/Users/wujie/PycharmProjects/Three-stage-stckelberg/game/Data/'

# 初始化限制车辆参数
Price_v, Q_total_m, f_m, k_m, e_m, Theta_m = [], [], [], [], [], []
Q_total_m_range = [10, 20]
# Q_total_m_range = [1, 3]

k_m_range = [1, 5]
e_m_range = [1, 5]
# s_m=1.75
s_m=0.5

############################################################################################################################################
# Userdevice 配置
bi_size_range = [10, 30]
bi_range = [bi_size_range[0], bi_size_range[1]]


############################################################################################################################################
# ECSP 配置
C_cloud_range = [0.2]
C_mec_range = [0.35, 0.8]
Q_ecsp_range = [300, 400]
e_cloud_range = [0.3]
e_mec_range = [0.1,0.5]
ecsp_beta_range=[0.1,0.3]
ecsp_beta,ecsp_cost, Q_ecsp, ecsp_enery=[],[],[],[]
############################################################################################################################################
class Vechicle:
    def __init__(self, index, Q_total_m, k_m, e_m, Theta_m, f_m):
        self.index = index
        self.Q_total_m = Q_total_m
        self.Theta_m = Theta_m
        self.f_m = f_m
        self.k_m = k_m
        self.e_m = e_m

    @staticmethod
    def build(file_name='vechicle.csv', cnt=v_number):
        """
        生成csv数据集文件
        随机生成csv数据集文件 Q_total_m,k_m,e_m,Theta_m
        :param name: 文件名称
        :param cnt: 循环次数
        """
        file_name = os.path.join(file_path, file_name)
        column_ranges = [(Q_total_m_range[0], Q_total_m_range[1]), (k_m_range[0], k_m_range[1]),
                         (e_m_range[0], e_m_range[1])]
        random_matrix = [[round(random.uniform(low, high), 2) for (low, high) in column_ranges] for i in
                         range(v_number)]
        for row in random_matrix:
            # row.append(round((row[1] * row[2]*s_m)**-1, 4))
            row.append((row[1] * row[2]*s_m)**-1)
        # 对每行按照最后一列的值进行升序排序(按照Theta升序)
        sorted_matrix = sorted(random_matrix, key=lambda x: x[-1])
        # 添加序号到每行的第一个位置
        sorted_matrix = [[i] + row for i, row in enumerate(sorted_matrix)]
        with open(file_name, 'w', newline="") as f:
            csv_writer = csv.writer(f)
            # 写入排序后的矩阵
            csv_writer.writerows(sorted_matrix)
            f.close()

    @staticmethod
    def read(count, file_name='vechicle.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        Q_total_m.clear()
        k_m.clear()
        e_m.clear()
        Theta_m.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    Q_total_m.append(float(ls[1]))
                    k_m.append(float(ls[2]))
                    e_m.append(float(ls[3]))
                    Theta_m.append(float(ls[4]))
                    c += 1
                else:
                    return ans
            return ans


class UserDevice:
    def __init__(self, index, bi, Q_total_v, f_v):
        self.index = index
        self.bi = bi

    @staticmethod
    def build(file_name='userDevice.csv', cnt=n_number):
        """
        生成csv数据集文件
        随机生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        file_name = os.path.join(file_path, file_name)
        with open(file_name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                ls.append(i)
                # bi = np.round(np.random.uniform(bi_range[0], bi_range[1]), 1)
                bi = round(random.uniform(bi_range[0], bi_range[1]), 2)
                ls.append(bi)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count, file_name='userDevice.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        bg.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    bg.append(float(ls[1]))
                    c += 1
                else:
                    return ans
            return ans


class ECSP:
    def __init__(self, index,ecsp_beta, ecsp_cost, Q_ecsp, ecsp_enery):
        self.index = index
        self.ecsp_beta = ecsp_beta
        self.ecsp_cost = ecsp_cost
        self.Q_ecsp = Q_ecsp
        self.ecsp_enery = ecsp_enery

    @staticmethod
    def build(file_name='ecsp.csv', cnt=ecsp_number):
        """
        生成csv数据集文件
        随机生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        file_name = os.path.join(file_path, file_name)
        with open(file_name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                ls.append(i)
                if i==0:
                    ecsp_beta = ecsp_beta_range[0]
                    ecsp_cost = np.round(np.random.uniform(C_cloud_range[0], C_cloud_range[0]), 1)
                    Q_ecsp = float("inf")
                    ecsp_enery = round(random.uniform(e_cloud_range[0], e_cloud_range[0]), 1)
                else:
                    ecsp_beta = ecsp_beta_range[1]
                    ecsp_cost = np.round(np.random.uniform(C_mec_range[0], C_mec_range[1]), 1)
                    Q_ecsp = round(random.uniform(Q_ecsp_range[0], Q_ecsp_range[1]))
                    ecsp_enery = round(random.uniform(e_mec_range[0], e_mec_range[1]), 2)
                ls.append(ecsp_beta)
                ls.append(ecsp_cost)
                ls.append(Q_ecsp)
                ls.append(ecsp_enery)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count, file_name='ecsp.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        ecsp_beta.clear()
        ecsp_cost.clear()
        Q_ecsp.clear()
        ecsp_enery.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    ecsp_beta.append(float(ls[1]))
                    ecsp_cost.append(float(ls[2]))
                    Q_ecsp.append(float(ls[3]))
                    ecsp_enery.append(float(ls[4]))
                    c += 1
                else:
                    return ans
            return ans


class LM:
    def __init__(self, index, Phi_m, Omega_m, Rho_m, Pi, Upsilon_j, Lambda_j):
        self.index = index
        self.Phi_m = Phi_m  # n
        self.Omega_m = Omega_m  # n
        self.Rho_m = Rho_m  # n+1
        self.Pi = Pi  # 1
        self.Upsilon_j = Upsilon_j  # 3
        self.Lambda_j = Lambda_j  # 3

    @staticmethod
    def build(file_name='LM.csv', cnt=v_number):
        """
        生成csv数据集文件
        随机生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        file_name = os.path.join(file_path, file_name)
        with open(file_name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                ls.append(i)
                Phi_m = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                Omega_m = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                Rho_m = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                ls.append(Phi_m)
                ls.append(Omega_m)
                ls.append(Rho_m)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count, file_name='LM.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        Phi_m.clear()
        Omega_m.clear()
        Rho_m.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    Phi_m.append(float(ls[1]))
                    Omega_m.append(float(ls[2]))
                    Rho_m.append(float(ls[2]))
                    c += 1
                else:
                    # Rho_m.append(float(1.0))
                    return ans
            return ans


if __name__ == '__main__':
    # UserDevice.build()
    # UserDevice.read(v_number)
    #
    # Vechicle.build()
    # Vechicle.read(v_number)
    #
    # LM.build()
    # LM.read(v_number)

    ECSP.build()
