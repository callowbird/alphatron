from utils import relu
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class alphatron(nn.Module):
    def __init__(self,x,d,kern):
        super(alphatron, self).__init__()
        self.anchor=Variable(x.clone().cuda(),requires_grad=False)
        self.d=d
        self.m=len(x)
        self.kern=kern
        self.alpha=nn.Linear(self.m,1,bias=False)
        self.alpha.weight.data.uniform_()
        self.alpha.weight.data*=10

    def forward(self, x):  #anchor is a minibatch of m points
        kernel1=self.kern(x,self.anchor,self.d)  #batch_size * m
        return self.alpha(kernel1).alphatronrelu()
        # return F.relu(self.alpha(kernel1))
    def predict(self,x):
        return self.forward(x)

