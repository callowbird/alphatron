import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes,m, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0,deg=1, kernel_type='mul'):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        block = BasicBlock
        # 1st conv before any dense block
        self.deg=deg
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.filters_1 = Variable(torch.randn(1, in_planes, 1, 1).fill_(1.0 / in_planes).cuda(),requires_grad=True)
        in_planes += m
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.filters_2 = Variable(torch.randn(1, in_planes, 1, 1).fill_(1.0 / in_planes).cuda(),requires_grad=True)
        in_planes += m
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.filters_3 = Variable(torch.randn(1, in_planes, 1, 1).fill_(1.0 / in_planes).cuda(),requires_grad=True)
        in_planes += m
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes
        self.kernel_type=kernel_type

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def step1(self,x):
        out= self.conv1(x)
        out = self.trans1(self.block1(out))
        return out

    def step2(self,x):
        out = self.trans2(self.block2(x))
        return out

    def step3(self,x):
        out = self.block3(x)
        return out

    def final(self,x):
        out = self.relu(self.bn1(x))
        out = F.avg_pool2d(out, 8)
        return out.view(-1,self.in_planes)

    def getFeature(self,inputs,anchor,filters):
        zip_a = F.conv2d(inputs, filters, padding=0).squeeze().unsqueeze(1).expand(inputs.size(0), anchor.size(0), inputs.size(2), inputs.size(3))
        zip_b = F.conv2d(anchor, filters, padding=0).squeeze().unsqueeze(0).expand(inputs.size(0), anchor.size(0), inputs.size(2), inputs.size(3))
        if self.kernel_type=='mul':
            return zip_a*zip_b
        elif self.kernel_type=='diff':
            return zip_a-zip_b
        elif self.kernel_type=='sqr':
            return (zip_a-zip_b)**2

    def combined_step(self,filter,x_in,a_in,step):
        x_o1=step(x_in)
        a_o1=step(a_in)

        x_f1=torch.cat([x_o1,self.getFeature(x_o1,a_o1,filter)],1)
        a_f1=torch.cat([a_o1,self.getFeature(a_o1,a_o1,filter)],1)
        return x_f1, a_f1


    def forward(self, x, anchor):  #anchor is a minibatch of m points

        x_f1,a_f1=self.combined_step(self.filters_1,x,anchor,self.step1)
        x_f2,a_f2=self.combined_step(self.filters_2,x_f1,a_f1,self.step2)
        x_f3,a_f3=self.combined_step(self.filters_3,x_f2,a_f2,self.step3)

        out=self.final(x_f3)

        return self.fc(out)
