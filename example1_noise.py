
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
from matplotlib import cm
from scipy.special import gamma
from torch.distributions.beta import Beta
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
from matplotlib import cm
from scipy.special import gamma
from torch.distributions.beta import Beta
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(888888)


# 模型搭建
class Net(nn.Module):
    def __init__(self,NN):# NL是有多少层隐藏层；#NN是每层的神经元数量
        super(Net,self).__init__()

        self.input_layer=nn.Linear(2,NN)

        self.hidden_layer1 = nn.Linear(NN, int(NN/2))
        self.hidden_layer2 = nn.Linear(int(NN/2), int(NN/4))

        self.hidden_layer3 = nn.Linear(int(NN/4), int(NN/4))
        self.hidden_layer4 = nn.Linear(int(NN/4), int(NN/4))
        self.hidden_layer5 = nn.Linear(int(NN/4), int(NN/4))


        self.output_layer=nn.Linear(int(NN/4),7)


        self.W1=  nn.Parameter(torch.tensor(0.5))
        self.W2 = nn.Parameter(torch.tensor(0.5))#损失函数上的权重


        self.register_parameter('W1', self.W1)
        self.register_parameter('W2', self.W2)




    def forward(self,x):#有两个特征的样本 x1的形状为（batch_size,2） [2000,2]

        out= torch.tanh(self.input_layer(x))

        out=torch.tanh( self.hidden_layer1(out))


        out=torch.tanh(self.hidden_layer2(out))
        out = torch.tanh( self.hidden_layer3(out))
        out = torch.tanh(self.hidden_layer4(out))
        out = torch.tanh( self.hidden_layer5(out))

        out_final = self.output_layer(out)

        return out_final


#lamda list
def talyer_cal_u(t,x,lamd): # (4000,7)


    u_pred = lamd[:, 0].unsqueeze(-1) + lamd[:, 1].unsqueeze(-1) * (
                lamd[:, 2].unsqueeze(-1) * x + lamd[:, 3].unsqueeze(-1) * t) + \
             lamd[:, 4].unsqueeze(-1) * (lamd[:, 5].unsqueeze(-1) * x + lamd[:, 6].unsqueeze(-1) * t) * (
                         lamd[:, 5].unsqueeze(-1) * x + lamd[:, 6].unsqueeze(-1) * t)

    return u_pred



def pde(x1, net, pt_x_collocation, pt_t_collocation):

    lamd= net(x1)  # (2000, 1)
    u=talyer_cal_u(pt_t_collocation,pt_x_collocation,lamd)


    u_tx = torch.autograd.grad(u, x1, grad_outputs=torch.ones_like(u),
                               create_graph=True, allow_unused=True)[0]  # u对x中的两个参数同时求导

    d_t = u_tx[:, 0].unsqueeze(-1)

    d_x = u_tx[:, 1].unsqueeze(-1)

    u_xx = torch.autograd.grad(d_x, x1, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 二阶偏导

    # 第二项
    alpha = torch.tensor([0.5])
    beta = torch.tensor([1.0])
    beta_dist = Beta(alpha, beta)

    # 生成随机样本
    samples = beta_dist.sample((4000,))  # 生成 2000 个样本

    tau = samples

    eples = 0.1
    t = 1 / pt_t_collocation
    t1 = eples = 0.01 * t

    # list
    result_list = []

    for i in range(4000):
        if tau[i] > t1[i]:
            result_list.append(tau[i])
        else:
            result_list.append(t1[i])

        # 将list转换为2000乘1的张量
    t_result_tensor = torch.tensor(result_list).reshape(4000, 1)
    lamd2=net(torch.cat((pt_t_collocation - t_result_tensor * pt_t_collocation, pt_x_collocation),1))
    # print("lamd2 shape:",lamd2.shape)

    u2=talyer_cal_u(pt_t_collocation - t_result_tensor * pt_t_collocation,pt_x_collocation,lamd2)

    sum=(u-u2) / t_result_tensor * pt_t_collocation

    lamd3=net(torch.cat((pt_zero_1, pt_x_collocation), 1))
    u3=talyer_cal_u(pt_zero_1,pt_x_collocation,lamd3)


    # result = ((1/4*(pt_t_collocation**0.8)) * sum) + ((u - u3) / pt_t_collocation**0.2)
    # result = (( 4 * (pt_t_collocation ** 0.2)) * sum) + ((u - u3) / pt_t_collocation ** 0.8)
    result = (torch.sqrt(pt_t_collocation) * sum) + ((u - u3) / torch.sqrt(pt_t_collocation))

    d_result = result
    return u_xx - (1/gamma(0.5))*d_result -d_x


net = Net(256)
# print(net)

mse_cost_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
# 初始化常量
# 初始条件t=0，x=1，x=0；
t_bc_zeros = np.zeros((100, 1))
x_in_pos_zero = np.zeros((100, 1))
x_in_neg_one = np.ones((100, 1))
x_in_neg_one1 = np.ones((4000, 1))
two_rig=2*x_in_neg_one1
pt_two_rig=Variable(torch.from_numpy(two_rig).float(), requires_grad=True)#矩阵two




t_in_var = np.random.uniform(low=0, high=1.0, size=(100, 1))  # t的取值范围
x_bc_var = np.random.uniform(low=0, high=1.0, size=(100, 1))  # x的取值范围
u_bc_sin = x_bc_var ** 2  # u=x**2，t=0时

# 将数据转化为pytorch可用
pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)  # 将x的取值范围转化
pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)  # 将t的取值范围转换
pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)  # 将t初始条件t=0转化
pt_u_bc_sin = Variable(torch.from_numpy(u_bc_sin).float(), requires_grad=False)  # 将t=0时，u等于x**2转化
pt_x_in_pos_zero = Variable(torch.from_numpy(x_in_pos_zero).float(), requires_grad=False)  # 边界条件x=0转化
pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)  # 边界条件x=1转化

zero_1 = np.zeros((4000, 1))

x_collocation = np.random.uniform(low=0.0, high=1.0, size=(4000, 1))  # PDE中的初始变量x
t_collocation = np.random.uniform(low=0.0, high=1.0, size=(4000, 1))  # PDE中的初始变量t
pt_zero_1 = Variable(torch.from_numpy(zero_1).float(), requires_grad=True)
pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
#
pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)


t = np.linspace(0, 1, 10)
x = np.linspace(0, 1, 10)
# exact=torch.square(x)+torch.square(t)
ms_t, ms_x = np.meshgrid(t, x)  # 将t和x组合起来
# print("ms_t",ms_t.shape)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
exact=torch.square(pt_x)+torch.square(pt_t)
# 迭代次数
start_time = time.time()
iterations = 9001
total_loss=[]
l2_erroer=[]
for epoch in range(iterations):
    optimizer.zero_grad()  # 梯度归0


    #求初始条件和边界条件的损失
    #init lamd
    noise_value = 0.001  # 噪音
    noise_value1 = torch.full((100, 1), noise_value)
    init_lamd = net(torch.cat([pt_t_bc_zeros, pt_x_bc_var], 1))  # 将x和t拼接在一起：其中t=0，x在取值范围内

    #bc1 lamd
    bc1_lamd = net(torch.cat([pt_t_in_var, pt_x_in_pos_zero], 1))  # 边界条件拼接在一起：其中t在取值范围内，x=0

    #bc2 lamd
    bc2_lamd = net(torch.cat([pt_t_in_var, pt_x_in_neg_one], 1))  # 边界条件拼接在一起：其中t在取值范围内，x=1

    # init out u
    init_u=talyer_cal_u(pt_t_bc_zeros,pt_x_bc_var,init_lamd)
    mse_u_2 = mse_cost_function(init_u, pt_u_bc_sin+noise_value1 )  # 初始条件mse

    #bc1 out u
    bc1_u = talyer_cal_u(pt_t_in_var,pt_x_in_pos_zero,bc1_lamd)
    mse_u_3 = mse_cost_function(bc1_u, (pt_t_in_var ** 2)+noise_value1 )  # 边界x=0时

    #bc2 out u
    bc2_u = talyer_cal_u(pt_t_in_var, pt_x_in_neg_one,bc2_lamd)
    mse_u_4 = mse_cost_function(bc2_u, (pt_x_in_neg_one + pt_t_in_var ** 2)+noise_value1 )  # 边界x=1时

#将x，t带入PDE公式 (2000，2)
    # print('torch.cat([pt_t_collocation, pt_x_collocation], 1)',torch.cat([pt_t_collocation, pt_x_collocation], 1).shape)
    f_out = pde(torch.cat([pt_t_collocation, pt_x_collocation], 1), net,pt_x_collocation,pt_t_collocation)
    value1=2.5
    value1_tenser = torch.full((4000, 1), value1)    #mse_f_1 = mse_cost_function(f_out, pt_two_rig+ 2*torch.square(pt_x_collocation)  + 2 *torch.square(pt_t_collocation) )
    mse_f_1 = mse_cost_function(f_out,
                                pt_two_rig - (2 * pt_x_collocation)- ((2 * pt_t_collocation**1.5)/gamma(value1)))

    # lamd_pt_u0 = net(torch.cat([pt_t, pt_x], 1))
    # # exact=torch.square(pt_x)+torch.square(pt_t)
    # pt_u0 = talyer_cal_u(pt_t, pt_x, lamd_pt_u0)
    # error_sum = 0
    # true_value_sum = 0
    #
    # for predict_value, true_value in zip(pt_u0, exact):
    # # print(type(true_value))
    #     error_sum += np.square(predict_value.detach().numpy() - true_value.detach().cpu().numpy())
    #     true_value_sum += np.square(true_value.detach().cpu().numpy())
    #
    # l2_normal_error = np.sqrt(error_sum) / np.sqrt(true_value_sum)
    # l2_erroer.append(l2_normal_error.item())
    # print("l2_normal_number :", l2_normal_error)

# 将误差损失加起来
    l = net.W1 * mse_f_1 + net.W2 * (mse_u_2 + mse_u_3 + mse_u_4)

    l.backward()  # 反向传播
    optimizer.step()
    # l = l.item()
    end_time = time.time()

    # 计算时间差
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    # if epoch == iterations-1:
         # torch.save(net.state_dict(),'./model_weight/example1_have_noise_0.5prediction.pth')

    with torch.autograd.no_grad():
        # if epoch % 100 == 0:
            # print(epoch, "Traning Loss:", l.data1.numpy(),"W1",net.W1,"W2",net.W2,"W3",net.W3,"W4",net.W4,"W5",net.W5)#"lambda_1",net.lambda_1
             print(epoch, "Traning Loss:", l.data.numpy())#"lambda_1",net.lambda_1
             total_loss.append(l.detach().cpu().numpy())
#log_losses = [torch.log(torch.tensor(l)).item() for l in total_loss]


#save loss:
# np.save('./np_weight/41shuju_two_total_loss.npy',total_loss)
# np.save('./np_weight/3daxiu_1Dl2error.npy', l2_erroer)#example1_l2_0.2
#np.save('./np_weight/0.8_1Dl2error.npy.npy', l2_erroer)#example1_0.8_noise_l2
#np.save('./np_weight/0.8_total_loss.npy',total_loss)#example1_0.8_noise_loss

#加载网络：
# model=Net(256)
# model.load_state_dict(torch.load('./model_weight/example1_have_noise_0.5prediction.pth'))



## 画图 ##
t = np.linspace(0, 1, 10)
x = np.linspace(0, 1, 10)
#exact=torch.square(x)+torch.square(t)
ms_t, ms_x = np.meshgrid(t, x)
#print("ms_t",ms_t.shape)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)

lamd_pt_u0 = model(torch.cat([pt_t, pt_x], 1))

pt_u0=talyer_cal_u(pt_t,pt_x,lamd_pt_u0)

#loss=pt_u0-exact
#plt.show(loss)
u = pt_u0.data.cpu().numpy()

pt_u0 = u.reshape(10, 10)



import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([0,2])
ax.set_xlim([0,1])
ax.set_ylim([0,1])

# ax.invert_zaxis()
ax.invert_xaxis()

ax.set_xticks(np.arange(0,1.5,0.5))
ax.set_yticks(np.arange(0,1.2,0.2))
ax.set_zticks(np.arange(0,2.5,0.5))

ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.003, antialiased=True)

ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')

ax.xaxis.pane.fill = False#面板不填充颜色
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.show()

#画误差图

# plt.subplot(1,2,1) #(1 ,1) (1,2)
exact=torch.square(pt_x)+torch.square(pt_t)
# print("exact:",exact)


pt_u0=torch.tensor(pt_u0)
pt_u0=torch.reshape(pt_u0,shape=(100,1))



error_sum=0
true_value_sum=0

for predict_value, true_value in zip(pt_u0,exact):

    # print(type(true_value))
    error_sum+=np.square(predict_value.detach().numpy() - true_value.detach().cpu().numpy())
    true_value_sum+=np.square(true_value.detach().cpu().numpy())




l2_normal_error=np.sqrt(error_sum) / np.sqrt(true_value_sum)
print("l2_normal_number :",l2_normal_error)

