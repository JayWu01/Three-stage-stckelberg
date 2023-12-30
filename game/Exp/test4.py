
my_list = [1, 2, 3, 4, 5]
m = 1

result_list = [sum([num for idx, num in enumerate(my_list) if idx != m]) for m in range(len(my_list))]

print(result_list)

# f_m_mine = [sum([num for idx, num in enumerate(my_list) if idx != m]) for m in range(len(my_list))]