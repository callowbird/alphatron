import torch
from torch.autograd import Variable
from torchvision import datasets, transforms


def getData(count,input_dim,network):
    x=Variable(torch.Tensor(count,input_dim).uniform_())
    y=network.forward(x)
    return {
       'x': x.data,
       'y': y.data.view(-1)
    }
def getPairedData(network,dataloader):
    network=network.cuda()
    x=torch.Tensor(1000,28*28)
    y=torch.Tensor(1000)
    counter=0
    for batch_idx, (data, target) in enumerate(dataloader):
        x[counter:counter+len(data)].copy_(data.view(-1,28*28))
        out=network.forward(Variable(data.cuda())).data.cpu().view(-1)
        y[counter:counter+len(data)].copy_(out)
        counter+=len(data)
        if counter>=1000:
            break
    return {
        'x': x,
        'y': y.view(-1)
    }

def mnistData(network):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/yang/data/data/MNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=1000, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/yang/data/data/MNIST', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=1000, shuffle=True)
    return getPairedData(network,train_loader),getPairedData(network,test_loader)

def evaluate(test, predictor, label="training data"):
    x=test['x']
    y=test['y']
    m=len(y)
    print("evaluating with "+str(m)+" data points ("+label+")")
    loss=0
    for i in range(m):
        loss+= (y[i] - predictor.predict(x[i])) ** 2
    print("loss=",loss/m)
def relu(a):
    if a>0:
        return a
    else:
        return 0