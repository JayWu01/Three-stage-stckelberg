import numpy as np

# 定义问题
n_var = 10
lb = 1
ub = 10

# 定义算法参数
population_size = 100
generations = 50
crossover_probability = 0.9
mutation_factor = 0.8

# 初始化种群
population = np.random.uniform(lb, ub, size=(population_size, n_var))

# 定义目标函数
def objective_function(x):
    return -np.sum(x)  # 最大化问题，加负号

# 定义约束条件
def constraint1(x):
    return np.all((x >= 1) & (x <= 10))

def constraint2(x):
    return np.sum(x) - 10

# 差分进化算法
for gen in range(generations):
    for i in range(population_size):
        # 随机选择三个个体作为变异向量
        indices = np.random.choice(population_size, 3, replace=False)
        a, b, c = population[indices]

        # 变异操作
        mutant = a + mutation_factor * (b - c)

        # 限制变异向量在合法范围内
        mutant = np.clip(mutant, lb, ub)

        # 交叉操作
        crossover_mask = np.random.rand(n_var) < crossover_probability
        trial_vector = np.where(crossover_mask, mutant, population[i])

        # 评估适应度
        if constraint1(trial_vector) and constraint2(trial_vector) <= 0:
            if objective_function(trial_vector) > objective_function(population[i]):
                population[i] = trial_vector

# 选择 Pareto 最优解
pareto_front = [ind for ind in population if constraint1(ind) and constraint2(ind) <= 0]
best_solution = max(pareto_front, key=objective_function)

print("Pareto Front:")
for ind in pareto_front:
    print(f"Objective: {objective_function(ind)}, Variables: {ind}")

print("\nBest Solution:")
print(f"Objective: {objective_function(best_solution)}, Variables: {best_solution}")
