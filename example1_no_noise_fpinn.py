
import time
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
epochs =9001   # 训练代数
h = 100      # 画图网格密度
N = 20    # 内点配置点数



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(888888)
# 划分网格


# Domain and Sampling
def interior(n=N):
    # 内点
    alp = 0.2
    x1=torch.linspace(0,1,n)
    t = torch.zeros_like(x1)
    for i in range(0, n):
        t[i] = (i+1) / n
    x = torch.zeros_like(x1)
    for i in range(0, n):
        x[i] = (i+1) / n
    X, T = torch.meshgrid(x, t)
    xt = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    xxx = xt[:, 0:1]
    ttt = xt[:, 1:2]
    #print(x,t)
    one=torch.ones_like(xxx)
    two=2*one
    cond= (2* ttt**(2-alp))/math.gamma(3-alp) +2*xxx-two
    return x.requires_grad_(True),t.requires_grad_(True),cond#,xxx.requires_grad_(True), ttt.requires_grad_(True), cond

def down(n=N):
    # 边界 u(x,0)=0
    x1=torch.rand(n,1)
    x=torch.zeros_like(x1)
    for i in range(0, n-1):
        x[i] = (i+1) / n
    t = torch.zeros_like(x1)
    cond = x**2
    return x.requires_grad_(True),t.requires_grad_(True), cond


def left(n=N):

    x1 = torch.rand(n, 1)
    t = torch.zeros_like(x1)
    for i in range(0, n):
        t[i] = (i+1) / n

    x = torch.zeros_like(x1)
    cond = t**2
    return x.requires_grad_(True), t.requires_grad_(True), cond

def right(n=N):

    x1 = torch.rand(n, 1)
    t = torch.zeros_like(x1)
    for i in range(0, n):
        t[i] = (i+1) / n
    x = torch.ones_like(x1)
    one=torch.ones_like(x1)
    cond =one+t**2
    return x.requires_grad_(True), t.requires_grad_(True), cond

# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2,256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),

            # torch.nn.Sigmoid(),
            # torch.nn.Linear(64, 1)
        )

        self.W1 = torch.nn.Parameter(torch.tensor(0.5))
        self.W2 = torch.nn.Parameter(torch.tensor(0.5))

        self.register_parameter('W1', self.W1)
        self.register_parameter('W2', self.W2)
    def forward(self, x):
        return self.net(x)


# Loss
# loss = torch.nn.MSELoss()
loss = torch.nn.MSELoss(reduction='mean')


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def fenshu(u):

    x, t, cond = interior()
    xx=x.detach().numpy()
    tt=t.detach().numpy()
    alp=0.2
    fu_star1 = torch.zeros([N**2, 1])
    X, T = torch.meshgrid(x, t)
    xt = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    xxx=xt[:,0:1]
    ttt=xt[:,1:2]

    uxy = u(torch.cat((xxx,ttt),dim=1))

    M=N
    d=1 / math.gamma(2 - alp)
    for i in range(1,M):
        sum1 = torch.zeros([N, 1])

        for j in range(1, i):
            # d=(np.exp((-na) * tt[i]) * (1 / 20) ** (1 - alp)) / math.gamma(2 - alp)
            sum1 += ((tt[i] - tt[j - 1]) ** (1 - alp) - (tt[i] - tt[j]) ** (1 - alp)) * ((uxy[M * j:M * (j + 1)] - uxy[M * (j - 1):M * (j)]) / (tt[j] - tt[j - 1]))


        if i == 1:
            fu_star1 = torch.concat([fu_star1[M*0:M*1], sum1], 0)
        else:
            fu_star1 = torch.concat([fu_star1,sum1], 0)
            #print(fu_star1.shape)
        c=gradients(uxy,xxx,2)

    return loss((d*fu_star1-c+gradients(uxy,xxx,1)),cond)


def l_down(u):

    x, t, cond = down()
    uxy = u(torch.cat((x, t), dim=1))
    return loss(uxy, cond)

def l_left(u):

    x, t, cond = left()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(uxy, cond)

def l_right(u):

    x, t, cond = right()
    uxy = u(torch.cat([x, t], dim=1))
    return loss(uxy, cond)


def l2_error(u):
    # l2误差
    x, t, cond =interior()
    xx = x.detach().numpy()
    tt = t.detach().numpy()
    # alp = 0.2
    X, T = torch.meshgrid(x, t)
    xt = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # xt = torch.from_numpy(xt1)
    xxx = xt[:, 0:1]
    ttt = xt[:, 1:2]
    # print('a',xxx)
    uxy = u(torch.cat((xxx, ttt), dim=1))
    u_xy = xxx**2+ttt**2
    return np.sqrt(torch.mean((u_xy - uxy) ** 2).detach()) / np.sqrt(torch.mean(u_xy ** 2).detach())
# Training
start_time = time.time()
u = MLP()
opt = torch.optim.Adam(params=u.parameters())
#opt = torch.optim.LBFGS(params=u.parameters())
min=10
l2_loss_list=[]
for i in range(epochs):
    opt.zero_grad()
    l = u.W1*fenshu(u) + u.W2*(l_down(u) + l_left(u) + l_right(u))
         # + l_data(u) \
    l.backward()
    opt.step()
    end_time = time.time()

    # 计算时间差
    execution_timei = end_time - start_time
    print("Execution time:", execution_timei, "seconds")
    # if  i== 9001:
         # torch.save(u.state_dict(),'./model_weight/time0.2.pth')


    if i % 1 == 0:
        l2_loss_list.append(l2_error(u).item())

#np.save('./np_weight/fpinn_0.2_daxiu_2Dl2error.npy', l2_loss_list)
# np.save('./np_weight/baqian_fpinn_0.2_daxiu_2Dl2error.npy', l2_loss_list)
# np.save('./np_weight/1D_baqian_fpinn_0.5_daxiu_2Dl2error.npy', l2_loss_list)

# print('min:', min, '  epo:', epochs)
#加载网络：
model=u
# model.load_state_dict(torch.load('./model_weight/model_xin41.pth'))

x=torch.linspace(0,1,20)
t=torch.linspace(0,1,20)
xx = x.detach().numpy()
tt = t.detach().numpy()
    # alp = 0.2
X, T = torch.meshgrid(x, t)
xt = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # xt = torch.from_numpy(xt1)
xxx = xt[:, 0:1]
ttt = xt[:, 1:2]
    # print('a',xxx)
start_inference_time = time.time()
uxy = model(torch.cat((xxx, ttt), dim=1))
end_inference_time = time.time()

inference_time = end_inference_time - start_inference_time
# u_p = u_pred.data.cpu().numpy()
print('inference_time',inference_time)




def draw_fig(l2_loss_list):

    plt.plot([i+1 for i in range(len(l2_loss_list))],l2_loss_list, '-', alpha=0.5, linewidth=1, label='\u03B1=0.5,\u03BB=-1')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    plt.yscale('log')
    plt.legend()  # 显示上面的label
    plt.xlabel('epoch')  # x_label
    plt.ylabel('$L^{2}$ error')  # y_label
    plt.xlim(1,len(l2_loss_list))
    # plt.savefig('D:\deepxde.example\examples\图表\l2_loss.jpg')
    plt.show()


if __name__ == '__main__':

    draw_fig(l2_loss_list)
