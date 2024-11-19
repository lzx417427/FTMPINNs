
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

import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.distributions.beta import Beta
# import differint.differint as df
import math
import sympy as sp
import numpy as np
import time
# import xlwt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

epochs = 3500    # 训练代数
h = 100    # 画图网格密度
N = 1000    # 内点配置点数
N1 = 100    # 边界点配置点数
N2 = 1000    # PDE数据点

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(12345)

# Domain and Sampling
def interior(n=N):
    #
    x = torch.rand(n, 1)
    z = torch.rand(n, 1)
    y = torch.rand(n, 1)
    one=torch.ones_like(x)
    two=2*one
    # cond = ((2/math.gamma(2.2))*y**1.2-(2*y**2)-two) * torch.exp(x+z)#pde 等式右边
    #cond = ((2 / math.gamma(2.5)) * y ** 1.5 - (2 * y ** 2) - two) * torch.exp(x + z)  # pde 等式右边
    cond = ((2 / math.gamma(2.8)) * y ** 1.8 - (2 * y ** 2) - two) * torch.exp(x + z)  # pde 等式右边

    #cond=(2*y**1.8/math.gamma(2.8))+2*x+0.8*x**2-two
    return x.requires_grad_(True), y.requires_grad_(True), z.requires_grad_(True),cond


def down_yy(n=N1):
    x = torch.rand(n, 1)
    z = torch.zeros_like(x)
    y = torch.rand(n, 1) #y=0
    one = torch.ones_like(x)
    noise = 0.001 * torch.ones_like(x)
    cond =(( (y) ** 2+one)*torch.exp(x))+noise
    return x.requires_grad_(True), y.requires_grad_(True), z.requires_grad_(True),cond


def up_yy(n=N1):
    x = torch.rand(n, 1)
    z=torch.ones_like(x)
    y = torch.rand(n, 1) #y=t
    one = torch.ones_like(x)
    noise = 0.001 * torch.ones_like(x)
    cond = (( y ** 2+one)*torch.exp(x+one))+noise
    return x.requires_grad_(True), y.requires_grad_(True),z.requires_grad_(True), cond


def down(n=N1):
    x = torch.rand(n, 1)
    z = torch.rand(n, 1)
    y = torch.zeros_like(x)#y=0
    noise = 0.001 * torch.ones_like(x)
    cond = (torch.exp(x+z))+noise
    # cond=x**2
    return x.requires_grad_(True), y.requires_grad_(True),z.requires_grad_(True), cond


def left(n=N1):

    y = torch.rand(n, 1)
    z = torch.rand(n, 1)
    x = torch.zeros_like(y)
    one = torch.ones_like(x)
    noise = 0.001 * torch.ones_like(x)
    cond = ((y**2+one)*torch.exp(z))+noise
    return x.requires_grad_(True), y.requires_grad_(True),z.requires_grad_(True), cond


def right(n=N1):

    y = torch.rand(n, 1)
    z = torch.rand(n, 1)
    x = torch.ones_like(y)
    one = torch.ones_like(x)
    noise = 0.001 * torch.ones_like(x)
    cond = ((y**2+one)*torch.exp(one+z))+noise
    # cond=one+y**2
    return x.requires_grad_(True), y.requires_grad_(True),z.requires_grad_(True), cond


# Neural Network
class MLP(torch.nn.Module):
    def __init__(self,AAF=True):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            # torch.nn.Linear(32, 32),
            # torch.nn.Tanh(),
            torch.nn.Linear(32, 5)
        )




        self.W1=  nn.Parameter(torch.tensor(0.5))
        self.W2 = nn.Parameter(torch.tensor(0.5))#损失函数上的权重
        # self.W3 = nn.Parameter(torch.tensor(0.2))
        self.register_parameter('W1', self.W1)
        self.register_parameter('W2', self.W2)



    def forward(self, x):


        return self.net(x)


# Loss
loss = torch.nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def talyer_cal_u(x, z, y, lamd):  # (4000,7)

    u_pred = lamd[:, 0].unsqueeze(-1) + lamd[:, 1].unsqueeze(-1) * (
            lamd[:, 2].unsqueeze(-1) * x + lamd[:, 3].unsqueeze(-1) * z + lamd[:, 4].unsqueeze(-1) * y)
    return u_pred
def fenshu(u):
    x,y,z,cond =interior()
    alpha = torch.tensor([0.8])
    beta = torch.tensor([1.0])
    beta_dist = Beta(alpha, beta)

    # 生成随机样本
    samples = beta_dist.sample((N,))  # 生成 2000 个样本
    # print(samples.shape)#[2000,1]
    tau = samples
    # print("tau:",tau)
    eples = 0.1
    t = 1 / y
    t1 = eples = 0.01 * t


    # 创建一个空的list
    result_list = []

    # 比较a和t的大小，并将较大的元素放入list中
    for i in range(N):
        if tau[i] > t1[i]:
            result_list.append(tau[i])
        else:
            result_list.append(t1[i])

        # 将list转换为2000乘1的张量
    t_result_tensor = torch.tensor(result_list).reshape(N, 1)
    # print((t_result_tensor * put_t).shape)
    lamd2=u(torch.cat(( x,z,(y - (t_result_tensor * y))),1))#z=y,y=t

    u2 = talyer_cal_u(x,z,(y - (t_result_tensor * y)),lamd2)
    # print("lamd2 shape:",lamd2.shape)

    lamd_uxy0 = u(torch.cat([x, z, y], dim=1))
    uxy =talyer_cal_u(x, z, y,lamd_uxy0)
    # u2=talyer_cal_u(z - t_result_tensor * z,x,y,lamd2)
    zero_1 = np.zeros((N, 1))
    pt_zero_1 = Variable(torch.from_numpy(zero_1).float(), requires_grad=True)
    sum=(uxy-u2) / t_result_tensor * y

    lamd3=u(torch.cat(( x,z,pt_zero_1), 1))
    u3=talyer_cal_u( x,z,pt_zero_1,lamd3)
    # result = (torch.sqrt(y) * sum) + ((uxy - u3) / torch.sqrt(y))
    result = ((1 / 4 * (y ** 0.8)) * sum) + ((uxy - u3) / y ** 0.2)

    # result = ((4*(y**0.2)) * sum) + ((uxy - u3) / y**0.8)
    d_result = result
     # u(x,y)

    return loss((1/gamma(0.8))*d_result -gradients(uxy, x, 2)-gradients(uxy, z, 2),cond)

def l_down_yy(u):
    # 损失函数L2
    x, y,z, cond = down_yy()

    lam3_uxy = u(torch.cat([x,z, y], dim=1))
    uxy = talyer_cal_u(x, z, y, lam3_uxy)
    return loss(uxy, cond)


def l_up_yy(u):
    # 损失函数L3
    x, y,z,cond = up_yy()
    lam4_uxy = u(torch.cat([x,z, y], dim=1))
    uxy = talyer_cal_u(x, z, y, lam4_uxy)
    return loss(uxy, cond)


def l_down(u):
    # 损失函数L4
    x, y,z, cond = down()
    lam5_uxy = u(torch.cat([x,z, y], dim=1))
    uxy = talyer_cal_u(x, z, y, lam5_uxy)
    return loss(uxy, cond)


def l_left(u):
    # 损失函数L6
    x, y,z, cond = left()
    lam6_uxy = u(torch.cat([x, z,y], dim=1))
    uxy = talyer_cal_u(x, z, y, lam6_uxy)
    return loss(uxy, cond)


def l_right(u):
    # 损失函数L7
    x, y,z, cond = right()
    lam7_uxy = u(torch.cat([x,z, y], dim=1))
    uxy = talyer_cal_u(x, z, y, lam7_uxy)
    return loss(uxy, cond)

def l2_error(u):

    xc = torch.linspace(0, 1, h)
    xm, ym, zm = torch.meshgrid(xc, xc, xc)

    xx = xm.reshape(-1, 1)
    yy = ym.reshape(-1, 1)
    zz = zm.reshape(-1, 1)
    lam8_uxy =  u(torch.cat((xx, zz, yy), dim=1))
    uxy = talyer_cal_u(xx, zz, yy, lam8_uxy)
    one=torch.ones_like(yy)
    u_xy = (yy*yy+one)*torch.exp(xx+zz)
    return torch.sqrt(torch.mean((u_xy - uxy) ** 2).detach()) / torch.sqrt(torch.mean(u_xy ** 2).detach())


# Training
start_time = time.time()
u = MLP()
opt = torch.optim.Adam(params=u.parameters())
best_loss = float('inf')
l2_loss_list=[]
for i in range(epochs):
    opt.zero_grad()
    l =u.W1* (fenshu(u)) \
        + u.W2*(l_down(u) + l_left(u) + l_right(u) +l_down_yy(u)+ l_up_yy(u))


    l.backward()
    opt.step()
    l = l.item()
    end_time = time.time()

# 计算时间差
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    # l2_loss_list.append(l2_error(u).item())
    if i % 50== 0:
        l2_loss_list.append(l2_error(u).item())

# print('min:', min, '  epo:', epochs)

    if l < best_loss:
         # torch.save(u.state_dict(), 'np_weight/example3_have_noise_0.5.pth')
         # torch.save(u.state_dict(), 'np_weight/example3_have_noise_0.8.pth')
         best_loss = l
    if i % 50 == 0:
        print(i, "l", l)
    # if i % 100 == 0:
    #     print(i,"l",l)
# np.save('./np_weight/noise_daxiu_2Dl2error.npy', l2_loss_list)
# np.save('./np_weight/example3_noise_0.8_2Dl2error.npy', l2_loss_list)
# Inference
xc = torch.linspace(0, 1, h)#h表示网格密度
xm, ym,zm = torch.meshgrid(xc, xc,xc)#xm 和ym 的网格大小都为（h，h）
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
zz=  zm.reshape(-1, 1)
xy = torch.cat([xx,zz, yy], dim=1)
lam_9_u_pred = u(xy)
u_pred=talyer_cal_u(xx, zz, yy, lam_9_u_pred)
# print("a",u_pred)

u_p = u_pred .data.cpu().numpy()

# u= u_pred.numpy()

pt_u0 = u_p.reshape(100, 100,100)

plt.imshow(pt_u0[0,:,:], cmap='hot',origin='lower',extent=[0, 1, 0, 1])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(' α=0.8,  Prediction: u(t,x)')
plt.show()
one=torch.ones_like(yy)
u_real = (yy*yy+one)*torch.exp(xx+zz)
print("b1",u_real)
u_error = torch.abs(u_pred-u_real)
print("u_error",u_error)

error_sum=0
true_value_sum=0
l2_value=[]
for predict_value, true_value in zip(u_pred,u_real):

    true_value_sum += np.square(true_value.detach().cpu().numpy())

    error_sum+=np.square(predict_value.detach().numpy() - true_value.detach().cpu().numpy())

    l2_normal_error=np.sqrt(error_sum) / np.sqrt(true_value_sum)
print("l2_normal_number :",l2_normal_error)

print("Min abs error is: ", float(torch.min(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx+zz)))))
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx+zz)))))
# print("Max abs error is: ", float(torch.max(torch.max(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx+zz))))/torch.max(torch.abs((yy*yy*yy+one)*torch.exp(xx+zz)))))
xc = torch.linspace(0, 1, h)#h表示网格密度
# print("b",xc)
xm, ym,zm = torch.meshgrid(xc, xc,xc)#xm 和ym 的网格大小都为（h，h）
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
zz=  zm.reshape(-1, 1)
xy = torch.cat([xx,zz, yy], dim=1)
lam_u_pred = u(xy)

u_pred= talyer_cal_u(xx,zz, yy, lam_u_pred)
error=torch.abs(u_pred-u_real)
error=error.data.cpu().numpy()
u_p = u_pred .data.cpu().numpy()
# print("a",u_pred)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(xx, zz, yy, c=u_p, cmap='viridis')
ax.invert_xaxis()
ax.set_xlabel('x')

ax.set_ylabel('y')
ax.set_zlabel('t')
fig.colorbar(img, ax=ax)
plt.title(r'$\alpha=0.8$, Prediction: $u(x,y,t)$')

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(xx, zz, yy, c=error, cmap='viridis',vmin=0.00,vmax=0.35)
ax.invert_xaxis()
ax.set_xlabel('x')

ax.set_ylabel('y')
ax.set_zlabel('t')
fig.colorbar(img, ax=ax)
plt.title(r'$\alpha=0.8$, abs error')

plt.show()
def draw_fig(l2_loss_list):

    plt.plot([i+1 for i in range(len(l2_loss_list))],l2_loss_list, '-', alpha=0.5, linewidth=1, label='\u03B1=0.5,\u03BB=-1')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.yscale('log')
    plt.legend()  # 显示上面的label
    plt.xlabel('epochs')  # x_label
    plt.ylabel('relative $l_{2}$ error')  # y_label
    plt.xlim(1,len(l2_loss_list))
    # plt.savefig('D:\deepxde.example\examples\图表\l2_loss.jpg')
    plt.show()


if __name__ == '__main__':

    draw_fig(l2_loss_list)

