import matplotlib.pyplot as plt
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [
    [33.487242176288525, 62.699335251304184, 90.5423243813811, 113.49689457861072, 132.08784397417617,
     154.4092008940024, 176.5356471855541, 201.13081615418938, 227.00918058348435, 239.64841625674418],
    [33.79715358638257, 62.5725283997306, 91.80954622237273, 115.47927833727186, 134.70453505219447, 157.78235579340617,
     177.35816889739533, 202.29905698318657, 232.31557675389396, 245.98314057279828],
    [30.70255582208295, 55.47885173470603, 79.227384122443, 106.04595464692619, 125.92503192158115, 137.84802615179146,
     163.05339025022676, 179.55012124747736, 197.16733729103044, 208.13678004040102]]

# 绘制折线图
plt.plot(range(len(socialWelfare[0])), socialWelfare[0], label='IA-C', marker='*', color='#d95319')
plt.plot(range(len(socialWelfare[1])), socialWelfare[1], label='NIA-S', marker='o')
plt.plot(range(len(socialWelfare[2])), socialWelfare[2], label='LP', marker='^', color='green')
# 添加图例
plt.legend()
plt.xlim(0, 9)
# 添加标题和轴标签
plt.xlabel('User Number', fontweight='bold', fontsize=15.5)
plt.ylabel('SocialWelfare', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(socialWelfare[0])), range(5, 55, 5), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()