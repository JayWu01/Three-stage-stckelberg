import random

# 生成10个均匀分布的概率值（小数点后最多两位）
probabilities = [round(random.uniform(0, 1), 2) for _ in range(10)]

# 确保概率值之和为1
total_probability = sum(probabilities)

# 计算归一化后的概率值，并保留小数点后两位
probabilities_normalized = [round(prob / total_probability, 2) for prob in probabilities]

print("Original Probabilities:", probabilities)
print("Normalized Probabilities (sum to 1):", probabilities_normalized)
