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


# parser = argparse.ArgumentParser(description='MNIST')
# parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
# parser.add_argument('--seed', type=int, default=30, metavar='S', help='random seed (default: 1)')
# parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
# parser.add_argument('--m', type=int, default=300, metavar='M', help='samples used per layer')
# parser.add_argument('--layers', type=int, default=2, metavar='L', help='how many layers')
# args = parser.parse_args()
class aa():
    pass
args=aa()
args.batch_size=64
args.seed=30
args.test_batch_size=1000
args.m=800
args.layers=2
args.lr=0.001
args.momentum=0
args.cuda=True
args.epochs=100
args.log_interval=30


torch.manual_seed(args.seed)
m=args.m

# here: input of MNIST data, make sure it makes sense


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.test_batch_size, shuffle=True)

input_dim=28*28
x=torch.Tensor(args.layers,m,input_dim).fill_(0)
counter_layer=0
counter_m=0
#fill x points
for batch_idx, (data, target) in enumerate(train_loader):
    for i in data:  # enumerate data points
        x[counter_layer][counter_m].copy_(i.resize_(input_dim))
        counter_m += 1
        if counter_m == args.m:
            counter_m = 0
            counter_layer += 1
            if counter_layer == args.layers:
                break
    if counter_layer == args.layers:
        break
x = Variable(x, requires_grad=False)
if args.cuda:
    x=x.cuda()


class Net(nn.Module):
    def __init__(self,deg):
        super(Net, self).__init__()

        self.relu=False
        self.linear1=nn.Linear(args.m,args.m)
        self.linear2=nn.Linear(args.m,args.m)
        self.bn1=nn.BatchNorm1d(args.m)
        self.bn2=nn.BatchNorm1d(args.m)
        self.deg=deg
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        # for layer in range(args.layers):
        #     curLinear=nn.Linear(args.m,args.m)
        #     setattr(self,"linear"+str(layer),curLinear)

        self.fc = nn.Linear(args.m, 10)
    def getkernel(self,a,b):
        inner=torch.mm(a,b.t())/1000
        ans=inner+0.001
        for i in range(2,self.deg+1):
            ans=ans+inner**i
        return ans

    def forward(self, input,x1,x2):
        #resize into one dim input

        kernel1=self.drop1(self.getkernel(input.view(input.size(0),input_dim),x1))  #batch_size * m

        if self.relu:
            output1=F.relu(self.bn1(self.linear1(kernel1)))
        else:
            output1=self.bn1(self.linear1(kernel1)).alphatronrelu()

        x1_kernel1=self.getkernel(x2,x1) #m*m

        if self.relu:
            x1_output1=F.relu(self.bn1(self.linear1(x1_kernel1)))
        else:
            x1_output1=self.bn1(self.linear1(x1_kernel1)).alphatronrelu()


        kernel2=self.drop2(self.getkernel(output1,x1_output1.t())) # batch_size * m

        if self.relu:
            output2=F.relu(self.bn2(self.linear2(kernel2)))
        else:
            output2=self.bn2(self.linear2(kernel2)).alphatronrelu()
        return F.log_softmax(self.fc(output2)) #batch_size*10

model=Net(1)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data,x[0],x[1])
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data,x[0],x[1])
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
    if epoch==60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

