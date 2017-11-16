import torch
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
