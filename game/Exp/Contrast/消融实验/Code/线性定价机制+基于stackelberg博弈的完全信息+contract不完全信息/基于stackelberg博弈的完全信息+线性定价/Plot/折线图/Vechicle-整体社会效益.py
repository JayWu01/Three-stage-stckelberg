import matplotlib.pyplot as plt
import numpy as np

# 四组数据
np.set_printoptions(precision=16)
socialWelfare = [[100.12796887802119, 106.19076668010068, 108.53256967558933, 109.1990348436584, 110.57933143826808,
                  111.92889329377228, 112.44745879798933, 111.80379317505611, 112.91576987801949, 113.48704372270964,
                  113.49689457861072, 113.68633508671272, 115.29664418564332],
                 [100.12796887802119, 106.19074056307332, 108.62293932900043, 109.49602946390058, 111.13651417199685,
                  112.46930597436842, 111.92044186675678, 112.83575817319243, 113.71838522341044, 114.5827439153567,
                  115.47927833727186, 116.38177135448406, 96.71941265932986],
                 [100.12796887802119, 103.72668668152207, 106.28626133938467, 109.32346354014297, 109.33956978766159,
                  100.08038357277965, 100.016890534361, 110.5087294225272, 100.1743315044348, 107.49552827799562,
                  106.04595464692619, 111.03448936778115, 105.2394387630879]]

# 绘制折线图
plt.plot(range(len(socialWelfare[0])), socialWelfare[0], label='IA-C', marker='*', color='#d95319')
plt.plot(range(len(socialWelfare[1])), socialWelfare[1], label='NIA-S', marker='o')
plt.plot(range(len(socialWelfare[2])), socialWelfare[2], label='LP', marker='^', color='green')
# 添加图例
plt.legend()
plt.xlim(0, 12)
# 添加标题和轴标签
plt.xlabel('Vechicle Number', fontweight='bold', fontsize=15.5)
plt.ylabel('SocialWelfare', fontweight='bold', fontsize=15.5)
plt.xticks(range(len(socialWelfare[0])), range(0, 13), fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形
plt.subplots_adjust(left=0.15)
plt.show()
