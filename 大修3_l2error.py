"""
A scratch for PINN solving the following PDE
u_xx-u_yyyy=(2-x^2)*exp(-y)
Author: ST
Date: 2023/2/26
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# import matplotlib
# matplotlib.use('TkAgg')  # 或其他您所需的交互式框架
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
from matplotlib import cm
from scipy.special import gamma
from torch.distributions.beta import Beta
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def setup_seed(seed):
    torch.manual_seed(seed)#每次运行代码时，我们都会得到相同的随机参数初始化
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(888888)

# two_total_loss=np.load('./np_weight/daxiu_2Dl2error.npy').tolist()
# five_total_loss=np.load('./np_weight/0.5_daxiu_2Dl2error.npy').tolist()
# two_total_loss=np.load('./np_weight/1daxiu_1Dl2error.npy').tolist()
# five_total_loss=np.load('./np_weight/2daxiu_1Dl2error.npy').tolist()

#图四
# plt.rcParams['font.size'] = 11
# two_total_loss=np.load('./np_weight/3daxiu_1Dl2error.npy').tolist()
# five_total_loss=np.load('./np_weight/4daxiu_1Dl2error.npy').tolist()
# eight_total_loss=np.load('./np_weight/example1_no_noise_0.8_l2error.npy').tolist()
# # # f_two_total_loss=np.load('./np_weight/fpinn_0.2_daxiu_2Dl2error.npy').tolist()
# # # f_five_total_loss=np.load('./np_weight/fpinn_0.5_daxiu_2Dl2error.npy').tolist()
# # f_five_total_loss=np.load('./np_weight/baqian_fpinn_0.5_daxiu_2Dl2error.npy').tolist()
# # f_two_total_loss=np.load('./np_weight/baqian_fpinn_0.2_daxiu_2Dl2error.npy').tolist()
# # f_eight_total_loss=np.load('./np_weight/example1_no_noise_0.8_fpinn_l2error.npy').tolist()
# f_five_total_loss=np.load('./np_weight/1D_baqian_fpinn_0.5_daxiu_2Dl2error.npy').tolist()
# f_two_total_loss=np.load('./np_weight/1D_0.5_baqian_fpinn_daxiu_2Dl2error.npy').tolist()
# f_eight_total_loss=np.load('./np_weight/1D_0.8_baqian_fpinn_daxiu_2Dl2error.npy').tolist()

#图10
# plt.rcParams['font.size'] = 14
# two_total_loss=np.load('./np_weight/1daxiu_1Dl2error.npy').tolist()
# five_total_loss=np.load('./np_weight/2daxiu_1Dl2error.npy').tolist()
# eight_total_loss=np.load('./np_weight/0.8_1Dl2error.npy.npy').tolist()


#图七
two_total_loss=np.load('./np_weight/daxiu_2Dl2error.npy').tolist()
five_total_loss=np.load('./np_weight/0.5_daxiu_2Dl2error.npy').tolist()
eight_total_loss=np.load('./np_weight/example3_no_noise_0.8_2Dl2error.npy').tolist()
# # f_two_total_loss=np.load('./np_weight/noise_fpinn_0.2_daxiu_1Dl2error.npy').tolist()
# # f_five_total_loss=np.load('./np_weight/noise_fpinn_0.5_daxiu_1Dl2error.npy').tolist()
# f_two_total_loss=np.load('./np_weight/2D_fpinn_0.2_daxiu_2Dl2error.npy').tolist()#0.2
f_two_total_loss=np.load('./np_weight/no_noise_2D_fpinn_0.2_daxiu_2Dl2error.npy').tolist()#
f_five_total_loss=np.load('./np_weight/no_noise_2D_fpinn_0.5_daxiu_2Dl2error.npy').tolist()#
f_eight_total_loss=np.load('./np_weight/no_noise_2D_fpinn_0.8_daxiu_2Dl2error.npy').tolist()

# f_five_total_loss=np.load('./np_weight/2D_fpinn_0.5_daxiu_2Dl2error.npy').tolist()#0.5
# f_noise_two_total_loss=np.load('./np_weight/noise_daxiu_2Dl2error.npy').tolist()
# f_noise_five_total_loss=np.load('./np_weight/noise_0.5_daxiu_2Dl2error.npy').tolist()
#f_two_total_loss=np.load('./np_weight/fpinn_0.2_daxiu_2Dl2error.npy').tolist()
#f_five_total_loss=np.load('./np_weight/fpinn_0.5_daxiu_2Dl2error.npy').tolist()
# f_eight_total_loss=np.load('./np_weight/example3_fpinn_2Dl2error_0.8.npy').tolist()#0.8
#图13
# plt.rcParams['font.size'] = 14
# f_noise_two_total_loss=np.load('./np_weight/noise_daxiu_2Dl2error.npy').tolist()
# f_noise_five_total_loss=np.load('./np_weight/noise_0.5_daxiu_2Dl2error.npy').tolist()
# f_noise_eight_total_loss=np.load('./np_weight/example3_noise_0.8_2Dl2error.npy').tolist()

#
#图表7
# sturcture150=np.load('./np_weight/SEN_0.2_41_daxiu_2Dl2error.npy').tolist()
# sturcture200=np.load('./np_weight/SEN_0.2_4_daxiu_2Dl2error.npy').tolist()
# sturcture258=np.load('./np_weight/SEN_0.2_43_daxiu_2Dl2error.npy').tolist()
# sturcture150_5=np.load('./np_weight/SEN_0.5_42_daxiu_2Dl2error.npy').tolist()
# sturcture200_5=np.load('./np_weight/SEN_0.5_41_daxiu_2Dl2error.npy').tolist()
# sturcture258_5=np.load('./np_weight/SEN_0.5_4_daxiu_2Dl2error.npy').tolist()
#


#图表六
# layers2=np.load('./np_weight/SEN_0.2_2_daxiu_2Dl2error.npy').tolist()
# layers4=np.load('./np_weight/SEN_0.2_44_daxiu_2Dl2error.npy').tolist()
# layers7=np.load('./np_weight/SEN_0.2_7_daxiu_2Dl2error.npy').tolist()
# layers2_5=np.load('./np_weight/SEN_0.5_2_daxiu_2Dl2error.npy').tolist()
# layers4_5=np.load('./np_weight/SEN_0.5_44_daxiu_2Dl2error.npy').tolist()
# layers7_5=np.load('./np_weight/SEN_0.5_7_daxiu_2Dl2error.npy').tolist()
#




# np.load('./np_weight/shenjingyuan0.2_1.npy',total_loss').tolist()
# example1=np.load('./np_weight/shenjingyuan0.2_1.npy').tolist()
# example2=np.load('./np_weight/shenjingyuan0.2_2.npy').tolist()
# example3=np.load('./np_weight/shenjingyuan0.2_3.npy').tolist()
# example4=np.load('./np_weight/shenjingyuan0.5_1.npy').tolist()
# example5=np.load('./np_weight/shenjingyuan0.5_2.npy').tolist()
# example6=np.load('./np_weight/shenjingyuan0.5_3.npy').tolist()

#图表4
# example1_net=np.load('./np_weight/shenjingyuan0.2_1.npy').tolist()
# example2_net=np.load('./np_weight/shenjingyuan0.2_1-3.npy').tolist()
# example3_net=np.load('./np_weight/shenjingyuan0.2_1-1.npy').tolist()
# example4_net=np.load('./np_weight/shenjingyuan0.5_1-2.npy').tolist()
# example5_net=np.load('./np_weight/shenjingyuan0.5_1-3.npy').tolist()
# example6_net=np.load('./np_weight/shenjingyuan0.5_1-1.npy').tolist()
#

#图表五
# example1_net=np.load('./np_weight/shenjingyuan0.2_1-1_12.npy').tolist()
# example2_net=np.load('./np_weight/shenjingyuan0.2_1-1_1.npy').tolist()
# example3_net=np.load('./np_weight/shenjingyuan0.2_1-1.npy').tolist()
# example5_net=np.load('./np_weight/shenjingyuan0.2_1.npy').tolist()
# example4_net=np.load('./np_weight/shenjingyuan0.5_1-2.npy').tolist()
#
# example6_net=np.load('./np_weight/shenjingyuan0.5_1-1.npy').tolist()

# np.save('./np_weight/shenjingyuan0.2_2.npy',total_loss)#128
# np.save('./np_weight/shenjingyuan0.2_3.npy',total_loss)#64




log_losses1 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in two_total_loss:
    log_loss1 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses1.append(log_loss1)
log_losses2 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in five_total_loss:
    log_loss2 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses2.append(log_loss2)
# print(log_losses2)
log_losses3 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in eight_total_loss:
    log_loss3 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses3.append(log_loss3)



log_losses4 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in f_two_total_loss:
    log_loss4 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses4.append(log_loss4)

log_losses5 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in f_five_total_loss:
    log_loss5 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses5.append(log_loss5)

log_losses6 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in f_eight_total_loss:
    log_loss6 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses6.append(log_loss6)



# plt.plot(log_losses3, label='FTM-PINNs α=0.2', color='r',  linewidth=2,  linestyle='--')
# # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses1, label='fPINNs α=0.2', color='k',   linewidth=2,linestyle='-')
# plt.plot(log_losses2, label='fPINNs α=0.5', color='b',   linewidth=2,linestyle='-.')
# plt.plot(log_losses1, label=r'$\alpha=0.2$: Net Structure1', color='r',  linewidth=1,  linestyle='--')
# # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Net Structure2', color='b',   linewidth=1,linestyle='--')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Net Structure3', color='k',   linewidth=1,linestyle='--')
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Net Structure1', color='r',   linewidth=1,linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Net Structure2', color='b',   linewidth=1,linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Net Structure3', color='k',   linewidth=1,linestyle='-')



# plt.plot(log_losses1, label='FTM-PINNs α=0.2', color='k',  linewidth=2,  linestyle='-')
# plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='b',   linewidth=2,linestyle='--')
# plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='b',   linewidth=2,linestyle='-.')
# plt.plot(log_losses1, label='FTM-PINNs α=0.2', color='r',   linewidth=2,linestyle='--')
# # # plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=2,linestyle='-')
# # # plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='b',   linewidth=2,linestyle='-')
# plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses3, label='fPINN α=0.2', color='k',  linewidth=2,  linestyle='-')
# plt.plot(log_losses4, label='fPINN α=0.5', color='b',   linewidth=2,linestyle='--')
# # plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='b',   linewidth=2,linestyle='-.')


# plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=3', color='r',  linewidth=2,  linestyle='--')
# # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=4', color='b',   linewidth=2,linestyle='--')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=5', color='k',   linewidth=2,linestyle='--')
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=3', color='r',   linewidth=2,linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=4', color='b',   linewidth=2,linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=5', color='k',   linewidth=2,linestyle='-')
#



# plt.rcParams['lines.markersize'] = 1
# plt.title(r'$\alpha=0.2$')
plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')



plt.show()

#图4
# plt.rcParams['font.size'] = 14
# plt.plot(log_losses3, label='fPINN α=0.2', color='k',  linewidth=1.5,  linestyle='-')
# plt.plot(log_losses4, label='fPINN α=0.5', color='b',   linewidth=1.5,linestyle='-.')
# plt.plot(log_losses1, label='FTM-PINNs α=0.2', color='r',   linewidth=1.5,linestyle='--')
# # # plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=2,linestyle='-')
# # # plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='b',   linewidth=2,linestyle='-')
# plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='y',   linewidth=1.5,linestyle='--')

#图七
# # plt.rcParams['font.size'] = 9.5
plt.plot(log_losses4, label='fPINN α=0.2', color='green',  linewidth=1.5,  linestyle='-')
plt.plot(log_losses5, label='fPINN α=0.5', color='k',   linewidth=1.5,linestyle='--')
plt.plot(log_losses6, label='fPINN α=0.8', color='orange',   linewidth=1.5,linestyle='-')

plt.plot(log_losses1, label='FTM-PINNs α=0.2', color='r',   linewidth=1.5,linestyle='-')
# # plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=2,linestyle='-')
# # plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='b',   linewidth=2,linestyle='-')
plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='y',   linewidth=1.5,linestyle='-')
plt.plot(log_losses3, label='FTM-PINNs α=0.8', color='b',   linewidth=1.5,linestyle='-')
#图10
# plt.plot(log_losses1, label='α=0.2', color='r',   linewidth=1.5,linestyle='-')
# # # plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=2,linestyle='-')
# # # plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='b',   linewidth=2,linestyle='-')
# plt.plot(log_losses2, label='α=0.5', color='y',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses3, label='α=0.8', color='b',   linewidth=1.5,linestyle='-')


#图13
# plt.plot(log_losses1, label='α=0.2', color='r',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses2, label='α=0.5', color='y',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses3, label='α=0.8', color='b',  linewidth=1.5,  linestyle='-')






#图表四
# plt.rcParams['font.size'] = 10.5
# plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=4', color='orange', linewidth=1.5, linestyle='-')
#
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=5', color='blue', linewidth=1.5, linestyle='-')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=6', color='green', linewidth=1.5,
#          linestyle='-')  # 修正标签中的Layers=5为Layers=7
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=4', color='red', linewidth=1.5, linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=5', color='purple', linewidth=1.5, linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=6', color='gray', linewidth=1.5,
#          linestyle='-')  # 修正标签中的Layers=5为Layers=7







#图表五
# plt.rcParams['font.size'] = 10.2
# plt.plot(log_losses1, label=r'$\alpha=0.2$: Net Structure1', color='orange',  linewidth=1,  linestyle='-')
# # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Net Structure2', color='b',   linewidth=1,linestyle='-')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Net Structure3', color='green',   linewidth=1,linestyle='-')
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Net Structure1', color='r',   linewidth=1,linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Net Structure2', color='purple',   linewidth=1,linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Net Structure3', color='gray',   linewidth=1,linestyle='-')


#图表6
# plt.rcParams['font.size'] = 10.5
# plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=3', color='orange',  linewidth=1.5,  linestyle='-')
# # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='-')
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=4', color='b',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=5', color='green',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=3', color='r',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=4', color='purple',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=5', color='gray',   linewidth=1.5,linestyle='-')
#
#




# plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=4', color='orange', linewidth=1.5, linestyle='-')
#
# plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=4', color='blue', linewidth=1.5, linestyle='-')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=7', color='green', linewidth=1.5,
#          linestyle='-')  # 修正标签中的Layers=5为Layers=7
# plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=3', color='red', linewidth=1.5, linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=4', color='purple', linewidth=1.5, linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=7', color='gray', linewidth=1.5,
#          linestyle='-')  # 修正标签中的Layers=5为Layers=7

# #图表7
# plt.rcParams['font.size'] = 10.5
# plt.plot(log_losses1, label=r'$\alpha=0.2$: 3+[150]+[80]+[40]+[20]+[20]+5', color='orange',  linewidth=1.5,  linestyle='-')
# # # plt.plot(log_losses4, label='FTM-PINNs α=0.5', color='y',   linewidth=2,linestyle='--')
# plt.plot(log_losses2, label=r'$\alpha=0.2$: 3+[200]+[100]+[60]+[25]+[25]+5', color='blue',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses3, label=r'$\alpha=0.2$: 3+[256]+[128]+[64]+[32]+[32]+5', color='green',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='purple',   linewidth=1.5,linestyle='-')
# plt.plot(log_losses6, label=r'$\alpha=0.5$: 3+[256]+[128]+[64]+[32]+[32]+5', color='gray',   linewidth=1.5,linestyle='-')

plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')



# 添加局部放大图
# 添加一个标记放大区域的矩形框（可选）

# 选择要放大的数据范围（例如，最后10个点）
#plt.rcParams['font.size'] = 13
# zoom_x = range(len(log_losses1) - 3, len(log_losses1))
# zoom_y1 = log_losses1[len(log_losses1) - 3:]
# zoom_y2 = log_losses2[len(log_losses2) - 3:]
# zoom_y3 = log_losses3[len(log_losses3) - 3:]
# zoom_y4 = log_losses4[len(log_losses4) - 3:]
# zoom_y5 = log_losses5[len(log_losses5) - 3:]
# zoom_y6 = log_losses6[len(log_losses6) - 3:]
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# # 创建插图
# axins = inset_axes(plt.gca(), width="55%", height="40%",
#                    bbox_to_anchor=(0.5, 0.2, 1, 1),
#                    bbox_transform=plt.gca().transAxes,
#                    loc='lower left',
#                    borderpad=0)
#
# # 在插图中绘制放大的数据
# axins.plot(zoom_x, zoom_y1,  color='orange', linewidth=1, linestyle='-')
#
# axins.plot(zoom_x, zoom_y2,  color='b', linewidth=1, linestyle='-')
# axins.plot(zoom_x, zoom_y3,  color='green', linewidth=1, linestyle='-')
# axins.plot(zoom_x, zoom_y4,  color='r', linewidth=1, linestyle='-')
# axins.plot(zoom_x, zoom_y5,  color='purple', linewidth=1, linestyle='-')
# axins.plot(zoom_x, zoom_y6,  color='gray', linewidth=1, linestyle='-')
#
# # 设置插图的y轴为对数刻度
# axins.set_yscale('log')
# axins.set_ylabel('relative $l_{2}$ error (zoomed)')
# axins.set_title('Zoomed Inset')
# axins.legend(loc='best')
#
# # 隐藏插图的x轴刻度
# axins.xaxis.set_ticks_position('none')
# axins.xaxis.set_tick_params(which='both', size=0)
# axins.yaxis.set_ticks_position('left')
#
# 创建局部放大的插图
axins = zoomed_inset_axes(plt.gca(), zoom=2, loc='upper right')  # zoom是放大倍数，loc是插图的位置
axins.plot(log_losses4, color='green', linewidth=1, linestyle='-')
axins.plot(log_losses5, color='k', linewidth=1, linestyle='--')
axins.plot(log_losses6, color='orange', linewidth=1., linestyle='-')

# 设置插图的x和y限制
x1, x2, y1, y2 = 60, 70, 0.12,0.2  # 根据你的数据调整这些值
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 隐藏插图的x轴和y轴刻度
axins.set_xticks([])
axins.set_yticks([])

# 在主图上标记插图的位置
mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")

# 显示图像
plt.show()
# # 显示图形
plt.show()
