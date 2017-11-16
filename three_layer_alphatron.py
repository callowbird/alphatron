import torch

from algorithms.alphatron_plain import alphatron_plain
from models.three_layer import three_layer
from utils import getData, evaluate
from kernels.MKd import MkdKern

m=200
input_dim=100
sl=three_layer(input_dim,input_dim)

train=getData(m,input_dim,sl)
test=getData(500,input_dim,sl)

d=1
T=1000

alphatron=alphatron_plain(train["x"], d, MkdKern)

alphatron.train(T, train, evaluate, 0.1)

evaluate(train, alphatron)
evaluate(test, alphatron, "test data")


