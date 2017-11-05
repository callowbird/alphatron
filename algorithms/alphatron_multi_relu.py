from utils import relu
import torch


class alphatron_multi_relu():
    def __init__(self, x, d, kern, reluCount):
        self.x = x
        self.d = d
        self.m = len(x)
        self.kern = kern
        self.alpha = torch.FloatTensor(reluCount,self.m).zero_()
        self.reluCount=reluCount

    def perReluPredict(self, ind, x_input):
        cumu = 0
        for i in range(self.m):
            cumu += self.alpha[ind][i] * self.kern(x_input, self.x[i], self.d)
        return relu(cumu)
    def predict(self,x_input):
        cumu=0
        for i in range(self.reluCount):
            cumu+=self.perReluPredict(i,x_input)
        return cumu

    def train(self, T, train, evaluate, eta=1.0):
        eta = eta / self.m
        x = train["x"]
        y = train["y"].clone()
        for r in range(self.reluCount):
            for t in range(T):
                if t % 100 == 0:
                    print("step ", t)
                    # evaluate(train, self)
                alpha_update = self.alpha[r].clone()
                for i in range(self.m):
                    incre = eta * (y[i] - self.perReluPredict(r,x[i]))
                    alpha_update[i] += incre
                self.alpha[r].copy_(alpha_update)
            for i in range(self.m):
                y[i]-= self.perReluPredict(r,x[i])
