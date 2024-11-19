
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


example1_net=np.load('./np_weight/shenjingyuan0.2_1.npy').tolist()
example2_net=np.load('./np_weight/shenjingyuan0.2_1-3.npy').tolist()
example3_net=np.load('./np_weight/shenjingyuan0.2_1-1.npy').tolist()
example4_net=np.load('./np_weight/shenjingyuan0.5_1-2.npy').tolist()
example5_net=np.load('./np_weight/shenjingyuan0.5_1-3.npy').tolist()
example6_net=np.load('./np_weight/shenjingyuan0.5_1-1.npy').tolist()




log_losses1 = []


for loss in example1_net:
    log_loss1 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses1.append(log_loss1)
log_losses2 = []


for loss in example2_net:
    log_loss2 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses2.append(log_loss2)
# print(log_losses2)
log_losses3 = []


for loss in example3_net:
    log_loss3 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses3.append(log_loss3)



log_losses4 = []

# 遍历 two_total_loss 列表，对每个数取对数，并将结果存储到 log_losses 列表中
for loss in example4_net:
    log_loss4 = loss # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses4.append(log_loss4)

log_losses5 = []


for loss in example5_net:
    log_loss5 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses5.append(log_loss5)

log_losses6 = []


for loss in example6_net:
    log_loss6 = loss  # 取对数
    # log_losses1.append(log_loss1)
    # log_loss1 = '{:.2e}'.format(loss)
    log_losses6.append(log_loss6)





#图表四
plt.rcParams['font.size'] = 10.5
plt.plot(log_losses1, label=r'$\alpha=0.2$: Layers=4', color='orange', linewidth=1.5, linestyle='-')

plt.plot(log_losses2, label=r'$\alpha=0.2$: Layers=5', color='blue', linewidth=1.5, linestyle='-')
plt.plot(log_losses3, label=r'$\alpha=0.2$: Layers=6', color='green', linewidth=1.5,
         linestyle='-')  # 修正标签中的Layers=5为Layers=7
plt.plot(log_losses4, label=r'$\alpha=0.5$: Layers=4', color='red', linewidth=1.5, linestyle='-')
plt.plot(log_losses5, label=r'$\alpha=0.5$: Layers=5', color='purple', linewidth=1.5, linestyle='-')
plt.plot(log_losses6, label=r'$\alpha=0.5$: Layers=6', color='gray', linewidth=1.5,
         linestyle='-')  # 修正标签中的Layers=5为Layers=7








plt.yscale('log')
plt.ylabel('relative $l_{2}$ error')
plt.legend(loc='best')
plt.xlabel('epochs')


plt.show()
