
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


#图表六
layers2=np.load('./np_weight/SEN_0.2_2_daxiu_2Dl2error.npy').tolist()
layers4=np.load('./np_weight/SEN_0.2_44_daxiu_2Dl2error.npy').tolist()
layers7=np.load('./np_weight/SEN_0.2_7_daxiu_2Dl2error.npy').tolist()
layers2_5=np.load('./np_weight/SEN_0.5_2_daxiu_2Dl2error.npy').tolist()
layers4_5=np.load('./np_weight/SEN_0.5_44_daxiu_2Dl2error.npy').tolist()
layers7_5=np.load('./np_weight/SEN_0.5_7_daxiu_2Dl2error.npy').tolist()





log_losses1 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers2:
    log_loss1 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses1.append(log_loss1)
log_losses2 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers4:
    log_loss2 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses2.append(log_loss2)
# print(log_losses2)
log_losses3 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers7:
    log_loss3 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses3.append(log_loss3)



log_losses4 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers2_5:
    log_loss4 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses4.append(log_loss4)

log_losses5 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers4_5:
    log_loss5 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses5.append(log_loss5)

log_losses6 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in layers7_5:
    log_loss6 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses6.append(log_loss6)





#图表6
plt.rcParams['font.size'] = 10.5
plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=3', color='orange',  linewidth=1.5,  linestyle='-')
plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=4', color='b',   linewidth=1.5,linestyle='-')
plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=5', color='green',   linewidth=1.5,linestyle='-')
plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=3', color='r',   linewidth=1.5,linestyle='-')
plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=4', color='purple',   linewidth=1.5,linestyle='-')
plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=5', color='gray',   linewidth=1.5,linestyle='-')
#

plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')



plt.show()
