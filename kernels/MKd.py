import torch
from torch.autograd import Variable
def MkdKern(x1, x2, d):
    inn=torch.dot(x1,x2)
    ans=0
    for deg in range(d+1):
        ans+= inn** deg
    return ans
def MkdKern_fast(a, b,d):
    inner = torch.mm(a, b.t()) / 1000
    ans = inner + 0.001
    for i in range(2, d+ 1):
        ans = ans + inner ** i
    return ans
# calculate paiwise distances
def distance(X, Y, sqrt=False):

    nX = X.size(0)
    nY = Y.size(0)
    M = torch.zeros(nX, nY)
    X = X.view(nX,-1).cuda()
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1).cuda()
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M.copy_(X2.expand(nX,nY) + Y2.expand(nY,nX).transpose(0,1) - 2*torch.mm(X,Y.transpose(0,1)))
    del X, X2, Y, Y2
    if sqrt:
        M = ((M+M.abs())/2).sqrt()

    return M
def Gaussian_fast(a, b,d):
    dist=- distance(a.data.cuda(),b.data.cuda())/500
    return Variable(torch.exp(dist).cuda())
