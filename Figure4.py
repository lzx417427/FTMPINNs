
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

#图四
plt.rcParams['font.size'] = 11
two_total_loss=np.load('./np_weight/3daxiu_1Dl2error.npy').tolist()
five_total_loss=np.load('./np_weight/4daxiu_1Dl2error.npy').tolist()
eight_total_loss=np.load('./np_weight/example1_no_noise_0.8_l2error.npy').tolist()
f_five_total_loss=np.load('./np_weight/1D_baqian_fpinn_0.5_daxiu_2Dl2error.npy').tolist()
f_two_total_loss=np.load('./np_weight/1D_0.5_baqian_fpinn_daxiu_2Dl2error.npy').tolist()
f_eight_total_loss=np.load('./np_weight/1D_0.8_baqian_fpinn_daxiu_2Dl2error.npy').tolist()



log_losses1 = []

for loss in two_total_loss:
    log_loss1 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses1.append(log_loss1)
log_losses2 = []

for loss in five_total_loss:
    log_loss2 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses2.append(log_loss2)
# print(log_losses2)
log_losses3 = []


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




#图四
# # plt.rcParams['font.size'] = 9.5
plt.plot(log_losses4, label='fPINN α=0.2', color='green',  linewidth=1.5,  linestyle='-')
plt.plot(log_losses5, label='fPINN α=0.5', color='k',   linewidth=1.5,linestyle='--')
plt.plot(log_losses6, label='fPINN α=0.8', color='orange',   linewidth=1.5,linestyle='-')

plt.plot(log_losses1, label='FTM-PINNs α=0.2', color='r',   linewidth=1.5,linestyle='-')
# # plt.plot(log_losses4, label=r'$\alpha=0.5$: 3+[150]+[80]+[40]+[20]+[20]+5', color='r',   linewidth=2,linestyle='-')
# # plt.plot(log_losses5, label=r'$\alpha=0.5$: 3+[200]+[100]+[60]+[25]+[25]+5', color='b',   linewidth=2,linestyle='-')
plt.plot(log_losses2, label='FTM-PINNs α=0.5', color='y',   linewidth=1.5,linestyle='-')
plt.plot(log_losses3, label='FTM-PINNs α=0.8', color='b',   linewidth=1.5,linestyle='-')

plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')




#
# 创建局部放大的插图
axins = zoomed_inset_axes(plt.gca(), zoom=2, loc='upper right')  # zoom是放大倍数，loc是插图的位置
axins.plot(log_losses4, color='green', linewidth=1, linestyle='-')
axins.plot(log_losses5, color='k', linewidth=1, linestyle='--')
axins.plot(log_losses6, color='orange', linewidth=1., linestyle='-')

# 设置插图的x和y限制
# x1, x2, y1, y2 = 60, 70, 0.12,0.2  # 根据你的数据调整这些值
x1, x2, y1, y2 = 8000, 9001, 0.25,0.35  # 根据你的数据调整这些值
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
