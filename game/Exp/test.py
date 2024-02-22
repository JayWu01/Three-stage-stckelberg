#导入库
import matplotlib.pyplot as plt
import numpy as np
#设定画布。dpi越大图越清晰，绘图时间越久
fig=plt.figure(dpi=300)
#导入数据
x=list(np.arange(1, 21))
y=np.random.randn(20)
#绘图命令
plt.plot(x, y, lw=4, ls='-', c='b', alpha=0.1)
plt.plot()
#show出图形
plt.show()
#保存图片
fig.savefig("画布")