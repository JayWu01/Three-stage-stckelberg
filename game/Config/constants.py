import numpy as np
import csv
import os

# # 常量
n_number = 1
v_number = 10
Alpha_e = 2
Alpha_c = 1
phi_e = 1
phi_c = 1
Epsilon_1 = 1.65
s_k = 1.4 # (0.8-1.4)
cpi = 1
f_i = 1
f_j = 12
f_c = 10
Q_total = 80
Error_value = 0.000001

# 变量 按照一定规则生成
D_i = []
data_size_range = [10, 15]
R_ij = 10 * 2.7
R_ic = 5 * 2.7
t_i_max = []
# t_i_range = [5, 7]
t_i_range = [2.5, 2.5]
# t_i_range = [3.5, 3.5]
max_iteration = 2000
# 初始化乘子 1*n_number的矩阵  （n_number代表用户数量）
lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9 = [], [], [], [], [], [], [], [], []
# 初始化 P_ij, P_ic
Eta_ij, Eta_ic, P_ij, P_ic = [], [], [], []
canshu = [1, 10]
# canshu = [0, 0]

p_ij_max, p_ij_min = Alpha_e * phi_e / Epsilon_1, Alpha_e * phi_e / (
        Epsilon_1 * (phi_e + 1))  # 根据Eta_ij_min=0、Eta_ij_max=1得到的
p_ic_max, p_ic_min = Alpha_c * phi_c / Epsilon_1, Alpha_c * phi_c / (
        Epsilon_1 * (phi_c + 1))  # 根据Eta_ij_min=0、Eta_ij_max=1得到的

# 边缘服务器、云服务器的成本
Q_j, Q_c = 0.4, 0.2

file_path = 'D:/江西理工大学/边缘计算/Three-stage-stckelberg/game/Data/'
# file_path = '/Users/wujie/PycharmProjects/subgradpy-master/edge/experiment/Data/'

# 初始化志愿者车辆(价格、容量、计算能力)参数
Price_v, Q_total_v, f_v = [], [], []
Q_total_v_range = [data_size_range[1], data_size_range[1] * 1.5]
f_v_range = [3, 5]
# 平均价格
p_v_max = (0.5 ** (v_number ** -1)) * Q_j
l_th = 5
# 志愿者车辆的平均价格
# v_p = 0.5 ** (1 / l_th) * Q_j
vp_min=0.1
v_p = (Q_j-vp_min) * np.exp(-0.1 * l_th) + vp_min


class Lamd:
    def __init__(self, index, lamda_1, lamda_2, lamda_3, lamda_4, lamda_5, lamda_6, lamda_7, lamda_8, lamda_9):
        self.index = index
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.lamda_3 = lamda_3
        self.lamda_4 = lamda_4
        self.lamda_5 = lamda_5
        self.lamda_6 = lamda_6
        self.lamda_7 = lamda_7
        self.lamda_8 = lamda_8
        self.lamda_9 = lamda_9

    @staticmethod
    def build(file_name='lamd.csv', cnt=n_number):
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
                lamda_1 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_2 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_3 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_4 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_5 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_6 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_7 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_8 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                lamda_9 = np.round(np.random.uniform(canshu[0], canshu[1]), 1)

                ls.append(lamda_1)
                ls.append(lamda_2)
                ls.append(lamda_3)
                ls.append(lamda_4)
                ls.append(lamda_5)
                ls.append(lamda_6)
                ls.append(lamda_7)
                ls.append(lamda_8)
                ls.append(lamda_9)

                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count, file_name='lamd.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        lamda_1.clear()
        lamda_2.clear()
        lamda_3.clear()
        lamda_4.clear()
        lamda_5.clear()
        lamda_6.clear()
        lamda_7.clear()
        lamda_8.clear()
        lamda_9.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    lamda_1.append(float(ls[1]))
                    lamda_2.append(float(ls[2]))
                    lamda_3.append(float(ls[3]))
                    lamda_4.append(float(ls[4]))
                    lamda_5.append(float(ls[5]))
                    lamda_6.append(float(ls[6]))
                    lamda_7.append(float(ls[7]))
                    lamda_8.append(float(ls[8]))
                    lamda_9.append(float(ls[9]))
                    c += 1
                else:
                    return ans
            return ans


class Task:
    def __init__(self, index, data_size, delay_limit):
        self.index = index
        self.data_size = data_size
        self.delay_limit = delay_limit

    @staticmethod
    def build(file_name='task.csv', cnt=n_number):
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
                data_size = np.round(np.random.uniform(data_size_range[0], data_size_range[1]), 1)
                t_i_max = np.round(np.random.uniform(t_i_range[0], t_i_range[1]), 1)
                ls.append(i)
                ls.append(data_size)
                ls.append(t_i_max)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def buildPrice(file_name='price.csv', cnt=n_number):
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
                # Price_ij = np.round(np.random.uniform(p_ij_min, p_ij_max), 1)
                # Price_ic = np.round(np.random.uniform(p_ic_min, p_ic_max), 1)
                Price_ij = np.round(np.random.uniform(1, 2), 1)
                Price_ic = np.round(np.random.uniform(1, 2), 1)
                ls.append(i)
                ls.append(Price_ij)
                ls.append(Price_ic)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count, file_name='task.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        D_i.clear()
        t_i_max.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    D_i.append(float(ls[1]))
                    t_i_max.append(float(ls[2]))
                    c += 1
                else:
                    return ans
            return ans

    @staticmethod
    def readPrice(count, file_name='price.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c = 0
        P_ij.clear()
        P_ic.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    P_ij.append(float(ls[1]))
                    P_ic.append(float(ls[2]))
                    c += 1
                else:
                    return ans
            return ans


class Vechicle:
    def __init__(self, index, Price_v, Q_total_v, f_v):
        self.index = index
        self.Price_v = Price_v
        self.Q_total_v = Q_total_v
        self.f_v = f_v

    @staticmethod
    def build(file_name='vechicle.csv', cnt=v_number):
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
                Q_total_v = np.round(np.random.uniform(Q_total_v_range[0], Q_total_v_range[1]), 1)
                f_v = np.round(np.random.uniform(f_v_range[0], f_v_range[1]), 1)
                Price_v = np.round(np.random.uniform(canshu[0], canshu[1]), 1)
                ls.append(Q_total_v)
                ls.append(f_v)
                ls.append(Price_v)
                writer.writerow(ls)
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
        Q_total_v.clear()
        f_v.clear()
        Price_v.clear()
        file_name = os.path.join(file_path, file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c < count:
                    ls = rows
                    Q_total_v.append(float(ls[1]))
                    f_v.append(float(ls[2]))
                    Price_v.append(float(ls[3]))
                    c += 1
                else:
                    return ans
            return ans


if __name__ == '__main__':
    Task.build()
    Task.read(n_number)
    Task.buildPrice()
    Task.readPrice(n_number)
    Lamd.build()
    Task.read(n_number)
    Lamd.read(n_number)
    Vechicle.build()
    Vechicle.read(v_number)
