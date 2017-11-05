from utils import relu
import torch

class alphatron_plain():
    def __init__(self,x,d,kern):
        self.x=x
        self.d=d
        self.m=len(x)
        self.kern=kern
        self.alpha=torch.FloatTensor(self.m).zero_()
    def predict(self,x_input):
        cumu=0
        for i in range(self.m):
            cumu+=self.alpha[i]*self.kern(x_input,self.x[i],self.d)
        return relu(cumu)
    def train(self, T, train, evaluate, eta=1.0):
        eta=eta/self.m
        x=train["x"]
        y=train["y"]
        for t in range(T):
            if t%100==0:
                print("step ",t,end=",")
                evaluate(train,self)
            alpha_update=self.alpha.clone()
            for i in range(self.m):
                incre=eta* (y[i]-self.predict(x[i]))
                alpha_update[i]+= incre
            self.alpha.copy_(alpha_update)
