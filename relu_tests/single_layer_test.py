import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


userelu=False

input_dim=5

a=nn.Linear(input_dim,1,bias=False)

a.weight.data.fill_(1)
a.weight.data[0][2]=3

input=Variable(torch.Tensor(2,5).fill_(3),requires_grad=True)
input.data[0].fill_(-5)


if userelu:
    b=F.relu(a.forward(input))
else:
    b=a.forward(input).alphatronrelu()
    #b=b.alphatronrelu()


grad=Variable(torch.Tensor(2,1).fill_(1))

b.backward(grad)

print(b)

print(input.grad)
