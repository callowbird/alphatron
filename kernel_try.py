import torch
import argparse

from algorithms.alphatron_plain import alphatron_plain
from models.double_layer import double_layer
from utils import getData, evaluate
from kernels.MKd import MkdKern
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class aa():
    pass
args=aa()
args.batch_size=64
# args.seed=30
args.test_batch_size=1000
args.m=800
args.layers=2
args.lr=0.001
args.momentum=0
args.cuda=True
args.epochs=100
args.log_interval=100


# torch.manual_seed(args.seed)
m=args.m

# here: input of MNIST data, make sure it makes sense


train_data=torch.load("cifar10/train.dat")
train_x=train_data['x']
train_y=train_data['y'].long()
test_data=torch.load("cifar10/val.dat")
test_x=test_data['x']
test_y=test_data['y'].long()

input_dim=456
anchor=torch.Tensor(args.layers, m, input_dim).fill_(0)
counter_layer=0
counter_m=0
indices=torch.randperm(train_x.size(0))
for i in range(args.layers):
    anchor[i].copy_(torch.index_select(train_x, 0, indices[i * m:i * m + m]))
if args.cuda:
    anchor=anchor.cuda()
anchor = Variable(anchor, requires_grad=False)


class Net(nn.Module):
    def __init__(self,deg):
        super(Net, self).__init__()

        self.relu=True
        self.linear1=nn.Linear(args.m,args.m)
        self.linear2=nn.Linear(args.m,args.m)
        self.bn1=nn.BatchNorm1d(args.m)
        self.bn2=nn.BatchNorm1d(args.m)
        self.deg=deg

        self.fc = nn.Linear(args.m, 10)
    def getkernel(self,a,b):
        inner=torch.mm(a,b.t())/1000
        ans=inner+0.001
        for i in range(2,self.deg+1):
            ans=ans+inner**i
        return ans

    def forward(self, input,x1,x2):
        #resize into one dim input

        kernel1=self.getkernel(input.view(input.size(0),input_dim),x1)  #batch_size * m

        output1=F.relu(self.bn1(self.linear1(kernel1)))

        x1_kernel1=self.getkernel(x2,x1) #m*m


        x1_output1=F.relu(self.bn1(self.linear1(x1_kernel1)))

        kernel2=torch.mm(output1,x1_output1.t()) # batch_size * m

        output2=F.relu(self.bn2(self.linear2(kernel2)))

        return F.log_softmax(self.fc(output2)) #batch_size*10

model=Net(1)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch,batchsize):
    model.train()
    indices=torch.randperm(train_x.size(0))
    for i in range(train_x.size(0)//batchsize):
        batch_idx=i*batchsize
        data=torch.index_select(train_x, 0, indices[i * batchsize:(i+1) * batchsize])
        target=torch.index_select(train_y, 0, indices[i * batchsize:(i+1) * batchsize])
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, anchor[0], anchor[1])
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx , train_x.size(0),
                100. * batch_idx / train_x.size(0), loss.data[0]))
def test(batchsize):
    model.eval()
    indices=torch.randperm(test_x.size(0))
    test_loss = 0
    correct = 0
    for i in range(test_x.size(0)//batchsize):
        data=torch.index_select(test_x, 0, indices[i * batchsize:(i+1) * batchsize])
        target=torch.index_select(test_y, 0, indices[i * batchsize:(i+1) * batchsize])
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data, anchor[0], anchor[1])
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= test_x.size(0)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_x.size(0),
        100. * correct / test_x.size(0)))


for epoch in range(1, args.epochs + 1):
    train(epoch,args.batch_size)
    test(args.test_batch_size)
    if epoch==60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr/10

