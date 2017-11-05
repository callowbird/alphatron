import torch
def MkdKern(x1, x2, d):
    inn=torch.dot(x1,x2)
    ans=0
    for deg in range(d+1):
        ans+= inn** deg
    return ans