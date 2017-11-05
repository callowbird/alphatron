import torch
from torch.autograd import Variable


def getData(count,input_dim,network):
    x=Variable(torch.Tensor(count,input_dim).uniform_())
    y=network.forward(x)
    return {
       'x': x.data,
       'y': y.data.view(-1)
    }
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