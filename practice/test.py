import pandas as pd
import torch
import numpy as np
from torch.nn import init
import torch.nn as nn
import torch.optim as optim
import random
import torch.utils.data as Data
torch.set_default_tensor_type(torch.DoubleTensor)
X_train=np.loadtxt(r'.\X_train.txt')
X_test=np.loadtxt(r'.\X_test.txt')
y_train=np.loadtxt(r'.\y_train.txt')
y_test=np.loadtxt(r'.\y_test.txt')
#模型输入标签数
num_inputs=4
#数据预处理

#定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature, n_out = 1):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, n_out)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
#转化为tensor类
X_train=torch.tensor(X_train)
y_train=torch.tensor(y_train)
X_train=X_train.float()
y_train=y_train.float()
X_test=torch.tensor(X_test)
X_test=X_train.float()
y_test=torch.tensor(y_test)
batch_size=10
#随机读取样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
for x,y in data_iter(batch_size,X_train,y_train):
    print(x,y)
    break
# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#定义损失函数
loss=nn.MSELoss()
#定义优化函数
optimizer = optim.SGD(net.parameters(), lr=0.0001)
#训练
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter(batch_size,X_train,y_train):
        output = net(X.to(torch.double))
        l = loss(output.to(torch.float32), y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))