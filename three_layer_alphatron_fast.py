import torch

from algorithms.alphatron import alphatron  as alphatron_class
from models.three_layer import three_layer
from utils import getData, evaluate
from models.double_layer import double_layer
from kernels.MKd import MkdKern_fast
import torch.backends.cudnn as cudnn
import torch.nn as nn

# torch.manual_seed(30)
m=1000
input_dim=300
sl=three_layer(input_dim,input_dim)

train=getData(m,input_dim,sl)
test=getData(m,input_dim,sl)

d=1
T=1000

model=alphatron_class(train["x"], d, MkdKern_fast)
# model=double_layer(input_dim,300)

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

model = model.cuda()

cudnn.benchmark = True

# define loss function (criterion) and pptimizer
criterion = nn.MSELoss()
criterion.cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.1, nesterov=False, weight_decay=0)

for epoch in range(100):
    indices=torch.randperm(m)
    batchsize=100
    iter=0
    tot_loss=0
    for i in range(m//batchsize):
        input=torch.index_select(train["x"],0,indices[iter:iter+batchsize])
        target=torch.index_select(train["y"],0,indices[iter:iter+batchsize])
        iter+=batchsize
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        tot_loss+=loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%10==0:
        print("tot_loss",tot_loss,end=",")



indices=torch.randperm(m)
iter=0
tot_loss=0
for i in range(m//batchsize):
    input=torch.index_select(test["x"],0,indices[iter:iter+batchsize])
    target=torch.index_select(test["y"],0,indices[iter:iter+batchsize])
    iter+=batchsize
    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)
    tot_loss+=loss.data[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("tot_loss",tot_loss)
