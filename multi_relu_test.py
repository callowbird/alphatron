import torch

from algorithms.alphatron_multi_relu import alphatron_multi_relu
from models.double_layer import double_layer
from utils import getData, evaluate
from kernels.MKd import MkdKern

torch.manual_seed(30)
m=300
input_dim=10
sl=double_layer(input_dim,10)

train=getData(m,input_dim,sl)
test=getData(500,input_dim,sl)
d=3
T=1000

alphatron=alphatron_multi_relu(train["x"], d, MkdKern,3)

alphatron.train(T, train, evaluate, 0.05)

evaluate(train, alphatron)
evaluate(test, alphatron, "test data")

