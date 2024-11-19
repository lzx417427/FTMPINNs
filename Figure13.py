
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(888888)
#图13
plt.rcParams['font.size'] = 14
f_noise_two_total_loss=np.load('./np_weight/noise_daxiu_2Dl2error.npy').tolist()
f_noise_five_total_loss=np.load('./np_weight/noise_0.5_daxiu_2Dl2error.npy').tolist()
f_noise_eight_total_loss=np.load('./np_weight/example3_noise_0.8_2Dl2error.npy').tolist()




log_losses1 = []


for loss in f_noise_two_total_loss:
    log_loss1 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses1.append(log_loss1)
log_losses2 = []


for loss in f_noise_five_total_loss:
    log_loss2 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses2.append(log_loss2)
# print(log_losses2)
log_losses3 = []


for loss in f_noise_eight_total_loss:
    log_loss3 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses3.append(log_loss3)



#图13
plt.plot(log_losses1, label='α=0.2', color='r',   linewidth=1.5,linestyle='-')
plt.plot(log_losses2, label='α=0.5', color='y',   linewidth=1.5,linestyle='-')
plt.plot(log_losses3, label='α=0.8', color='b',  linewidth=1.5,  linestyle='-')






plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')



plt.show()
