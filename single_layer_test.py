import torch

from algorithms.alphatron_plain import alphatron_plain
from models.single_layer import single_layer
from utils import getData, evaluate
from kernels.MKd import MkdKern

m=100
input_dim=10
sl=single_layer(input_dim)

train=getData(m,input_dim,sl)
test=getData(500,input_dim,sl)

d=1
T=1000

alphatron=alphatron_plain(train["x"], d, MkdKern)

alphatron.train(T, train, evaluate, 0.1)

evaluate(train, alphatron)
evaluate(test, alphatron, "test data")


w=torch.Tensor(input_dim).zero_()
for i in range(m):
    for j in range(input_dim):
        w[j]+=alphatron.alpha[i]*train["x"][i][j]
print("w=",w.view(1,-1))
print("target=",sl.a.weight.data.view(1,-1))
print("diff=",torch.norm(w-sl.a.weight.data.view(-1)))
