import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size":16}
matplotlib.rc('font', **font)  # 一次定义终身使用
# 四组数据

# F_0_v,F_1_v,F_2_v =[1.4849237331657967, 1.4325095578739844, 1.3817820122717257, 1.332386902932578, 1.2842917944357335, 1.2374654186781917, 1.1918776196093344, 1.1474993011946748, 1.1043023783851253, 1.0622597308859745, 1.0213451595360261, 0.9815333451221437, 0.942799809467935, 0.9051208786476451, 0.868473648187552, 0.8328359501274811, 0.7981863218244504, 0.7645039763891068, 0.7317687746535498, 0.7111070319096868, 0.6909273840896583, 0.6713379642140037, 0.6523270348896263, 0.6338832038004198, 0.6159954103405134, 0.5986529128878686, 0.5906401484766494, 0.5827675191932945, 0.5751497230278846, 0.5677820734721625, 0.5606600114505012, 0.553779100747543, 0.5471350236390146, 0.5407235767149663, 0.5345406668853718, 0.5374060255073252, 0.531818564708152, 0.5263000126072243, 0.5209946445879631, 0.5158988650525436, 0.5160008347598913] , [-17.762670486294752, -16.77754063865192, -15.457038588014683, -14.210069487096114, -13.032748659734237, -11.921401061725094, -10.872577626357689, -9.883063460113165, -8.9497661197668, -8.069848102605901, -7.240616222247054, -6.459553342993937, -5.724232909828822, -5.03242248561293, -4.381998509658462, -3.770971541071643, -3.1974197342459583, -2.659570393351861, -2.1557340329049293, -1.6821532430408546, -1.2380294406551329, -0.8239064862973144, -0.43841199166612554, -0.0802178588924054, 0.25190765723485153, 0.5591339675934459, 0.8434657001899466, 1.0935576859590062, 1.3227081507943677, 1.5318280076180892, 1.7218013142325215, 1.8934519953256181, 2.047569162150414, 2.1849030356180825, 2.3051471878618455, 2.407809183239617, 2.4925035410658287, 2.5674875146046476, 2.6292889314188894, 2.67845601001048, 2.71850912844136] , [0.4632498057454313, 0.9120899610438371, 1.3541679890902238, 1.765451370754898, 2.14760131365607, 2.502175513641034, 2.8306342900533212, 3.134346528591995, 3.4146005582416925, 3.6726034406994237, 3.9094908989573227, 4.126330382360957, 4.324129060966101, 4.503832989184948, 4.6663344748508715, 4.812474212088353, 4.943047239693984, 5.058802120875935, 5.160446448260212, 5.248261565289773, 5.323167974988512, 5.385650435857665, 5.43628531166012, 5.47561944047242, 5.5041696912511595, 5.5224259579287, 5.531332923111854, 5.534223633003084, 5.537381489450655, 5.540789382067908, 5.544435382472774, 5.548304536724428, 5.552383907241638, 5.556661046156692, 5.560630389844091, 5.564342462362871, 5.5559353385761465, 5.55917247058059, 5.562438835578354, 5.5657347405143005, 5.5679859208001945] , [5.006705062458904, 5.265142701490141, 5.501495999999995, 5.716980643194631, 5.91273384889946, 6.089820235157494, 6.249237188208612, 6.391919778546708, 6.518745267712272, 6.630537244021664, 6.728069421487599, 6.812069132685263, 6.883220543209874, 6.942167612607171, 6.989516824196593, 7.0258397040120215, 7.051675147125395, 7.067531567867035, 7.073888888888888, 7.076313580246914, 7.078735802469135, 7.081155555555556, 7.083572839506173, 7.085987654320986, 7.0884, 7.090809876543209, 7.092013888888889, 7.093217283950618, 7.094420061728394, 7.095622222222222, 7.096823765432097, 7.098024691358026, 7.099225, 7.1004246913580245, 7.101632675627818, 7.10172805841986, 7.102901046520433, 7.104287348310274, 7.105778280960643, 7.10736887578678, 7.107815445935952] , [7.271073418993823, 7.271073418993823, 6.896815759310228, 6.541547440445442, 6.203863758979632, 5.882821775865766, 5.577537996072701, 5.28689838292059, 5.010695779541162, 4.7479068390049655, 4.497843443080962, 4.259639866502292, 4.033133022940316, 3.8175224674073145, 3.6122665003117227, 3.416682593807903, 3.2306409780186196, 3.0535123779103426, 2.8848676419567143, 2.724157714765599, 2.659950718634616, 2.598956567327325, 2.540914439586749, 2.4859097588299006, 2.433699480132975, 2.384068578926085, 2.337072249530996, 2.2319525011147086, 2.1345600826181386, 2.044436693387338, 1.9613846014007819, 1.8850031971053074, 1.8149819416551942, 1.7510172289573946, 1.6768939124317779, 1.5757523295236973, 1.4994722815859431, 1.413087076262248, 1.3332160373443895, 1.2596389146622422, 1.1919415043468355]
F_0_v = [0.8000000000000054, 0.8780487804878103, 0.918032786885252, 0.9586776859504194, 0.9586776859504194, 0.9581267217630915, 0.9581267217630915, 0.9983333333333385, 0.9983333333333385, 0.997777777777783, 0.9972222222222275]
F_1_v = [1.7142857142857184, 1.7142857142857184, 1.7142857142857184, 1.7142857142857184, 1.7142857142857184, 1.7685897435897475, 1.7685897435897475, 1.8233009708737895, 1.8233009708737895, 1.822653721682851, 1.879084967320265]
F_2_v = [2.2500000000000044, 2.2500000000000044, 2.2500000000000044, 2.250000000000004, 2.250000000000004, 2.2493055555555594, 2.2493055555555594, 2.247916666666671, 2.247916666666671, 2.3129824561403542, 2.312280701754389]

f_j_vop_v = np.array([
[3.1730955770692404e-05, 0.0, 0.0],
[0.09530429033513543, 0.0, 0.0],
[0.1639687138501872, 0.0, 0.0],
[0.20309698642849405, 0.02841728750271244, 0.0],
[0.21831930672035316, 0.07408424837828975, 0.0],
[0.2269184220710566, 0.1278600553634064, 0.0],
[0.23949163776952165, 0.1655797024588015, 0.0016603256901670527],
[0.26260116572158865, 0.2019539826016532, 0.015057547653270387],
[0.2693721888895051, 0.22226705210540199, 0.04679671875287772],
[0.2744388470262529, 0.23797673525351004, 0.1043816568390592],
[0.2792518680641922, 0.2814647545193676, 0.12789389909474957]
])


f_0_vop_v,f_1_vop_v,f_2_vop_v=f_j_vop_v[:, 0],f_j_vop_v[:, 1],f_j_vop_v[:, 2]

# 绘制折线图
plt.plot(range(len(F_0_v)), F_0_v, label='xx', marker='.')
plt.plot(range(len(F_1_v)), F_1_v, label='xx', marker='o')
plt.plot(range(len(F_2_v)), F_2_v, label='xx', marker='s')
plt.plot(range(len(f_0_vop_v)), f_0_vop_v, label='xx', marker='^')
plt.plot(range(len(f_1_vop_v)), f_1_vop_v, label='xx', marker='x')
plt.plot(range(len(f_2_vop_v)), f_2_vop_v, label='xx', marker='x')
# 添加图例
plt.legend()

# 添加标题和轴标签
plt.xlabel('xx', fontsize=16)
plt.ylabel('xx', fontsize=16)
plt.xticks(fontsize=13.5)  # 修改x轴刻度字体大小
plt.yticks(fontsize=13.5)  # 修改y轴刻度字体大小
# 添加图例并设置字体大小
plt.legend(fontsize='16')
# 显示图形

plt.show()
