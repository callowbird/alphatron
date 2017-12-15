import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def distance(new_f, anchor_f, sqrt):
    # new_f:     B*Ch*R*C
    # anchor_f:  A*Ch*R*C
    # target:    B*A* R*C
    B = new_f.size(0)
    A = anchor_f.size(0)
    Ch = new_f.size(1)
    R = new_f.size(2)
    C = new_f.size(3)
    X2 = (new_f * new_f).sum(1).unsqueeze(1).expand(B, A, R, C)  # B*A*R*C
    Y2 = (anchor_f * anchor_f).sum(1).unsqueeze(0).expand(B, A, R, C)  # B*A*R*C

    new_f = new_f.unsqueeze(1)  # becomes B*1*Ch*R*C
    anchor_f = anchor_f.unsqueeze(0)  # becomes 1*A*Ch*R*C
    new_f = new_f.expand(B, A, Ch, R, C)  # becomes B*A*Ch*R*C
    anchor_f = anchor_f.expand_as(new_f)  # becomes B*A*Ch*R*C
    inner = torch.sum(new_f * anchor_f, 2)  # get B*A*R*C
    M = X2 + Y2 - 2 * inner
    del X2, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def poly_kernel(new_f,anchor_f,deg):
    #new_f:     B*Ch*R*C
    #anchor_f:  A*Ch*R*C
    #target:    B*A* R*C
    target=new_f.new(new_f.size(0),anchor_f.size(0),new_f.size(2),new_f.size(3)).fill_(1)
    new_f=new_f.unsqueeze(1)  #becomes B*1*Ch*R*C
    anchor_f=anchor_f.unsqueeze(0) #becomes 1*A*Ch*R*C
    new_f=new_f.expand(new_f.size(0),anchor_f.size(1),new_f.size(2),new_f.size(3),new_f.size(4))
    anchor_f=anchor_f.expand_as(new_f)
    inner=torch.sum(new_f*anchor_f,2)  #get B*A*R*C
    for i in range(1,deg+1):
        target=target+inner**i  #target += inner^d
    return target

def gaussian_kernel(new_f,anchor_f,sigma):
    #exp(- dist/2/sigma)
    target=distance(new_f,anchor_f,sqrt=False)/2/sigma
    return torch.exp(-target)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate, turnOn, kernel, kernelPara, kernelSizeA,listOut):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        if turnOn:
            in_planes+=kernelSizeA
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.turnOn=turnOn
        self.kernel=kernel
        self.kernelPara=kernelPara
        self.listOut=listOut
        self.state=False

    def set_vector(self, vec):
        self.vec=vec.detach()   #vec has size A*Ch*R*C
    def set_state(self):
        self.state=True

    def forward(self, x):
        in_x=x
        if self.turnOn:  #if turned on, we need to update extra feature.. else do nothing.
            new_f=x[-1]  #get new feature, has size B*Ch*R*C
            if self.state:  #Oh! The current state is on, which means we need to save the current feature
                self.set_vector(new_f) #out has size A*Ch*R*C
                self.state=False
                return [] #stop forward here.
            if self.kernel=='gaussian':
                extra_f=gaussian_kernel(new_f,self.vec,self.kernelPara)  #get size B*A*R*C
            else:
                extra_f=poly_kernel(new_f,self.vec,self.kernelPara)  #get size B*A*R*C
            in_x=torch.cat(x,1)
            x=torch.cat(x+[extra_f],1)
        else:
            in_x=x
        tmp=self.bn1(x)
        out = self.conv1(self.relu(tmp))

        # out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.turnOn or self.listOut:
            return [in_x,out]
        else:
            return torch.cat([in_x,out],1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, listIn=True):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.listIn=listIn
    def forward(self, x):
        if isinstance(x, list) and len(x)==0:
            return []
        if self.listIn:
            x=torch.cat(x,1)
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate,kernel,kernelPara,kernelSizeA,turnOn):
        super(DenseBlock, self).__init__()
        self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate, kernel, kernelPara,
                                      kernelSizeA,turnOn)
        self.layer=nn.Sequential(*self.layers)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate,kernel,
                    kernelPara,kernelSizeA,turnOn):
        self.layers = []
        self.count=0
        for i in range(int(nb_layers)):
            if i==0:  #first layer, should not turn on
                self.layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate,turnOn=False,listOut=turnOn,
                     kernel=kernel,kernelPara=kernelPara,kernelSizeA=kernelSizeA))
            else:
                self.layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate,turnOn=turnOn,listOut=turnOn,
                     kernel=kernel,kernelPara=kernelPara,kernelSizeA=kernelSizeA))
                self.count+=1
    def set_state(self):
        for li in self.layers:
            li.set_state()
    def forward(self, x):
        for li in self.layers:
            if isinstance(x, list) and len(x)==0:
                return [] #stop here.
            x=li.forward(x)
        return x

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5,  dropRate=0.0, kernel='gaussian', kernelPara=1,kernelSizeA=50,turnOn=True):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3

        n = n/2
        block = BottleneckBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate,kernel,kernelPara,kernelSizeA,turnOn)

        in_planes = int(in_planes+n*growth_rate)

        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,listIn=turnOn)

        in_planes = int(math.floor(in_planes*reduction))

        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate,kernel,kernelPara,kernelSizeA,turnOn)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate,listIn=turnOn)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate,kernel,kernelPara,kernelSizeA,turnOn)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes
        self.needBatch=self.block1.count+self.block2.count+self.block3.count
        self.turnOn=turnOn

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def set_state(self):
        self.block1.set_state()
        self.block2.set_state()
        self.block3.set_state()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        if isinstance(out, list) and len(out)==0:
            return []
        out = self.trans2(self.block2(out))
        if isinstance(out, list) and len(out)==0:
            return []
        out=self.block3(out)
        if isinstance(out, list) and len(out)==0:
            return []
        if self.turnOn:
            out = torch.cat(out,1)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
