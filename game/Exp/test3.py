import random

def create(v_number):
    global lamda_m
    # 生成10个均匀分布的概率值（小数点后最多两位）
    probabilities = [round(random.uniform(0, 1), 2) for _ in range(v_number)]

    # 确保概率值之和为1
    total_probability = sum(probabilities)

    # 计算归一化后的概率值，并保留小数点后两位
    lamda_m = [round(prob / total_probability, 2) for prob in probabilities]

    return lamda_m


if __name__ == '__main__':
    result=[]
    for i in range(1,11):
        ls=create(i)
        result.append(ls)
        # result[i]=ls

    print(result)