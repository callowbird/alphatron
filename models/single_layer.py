import torch.nn as nn
import torch.nn.functional as F

class single_layer(nn.Module):
    def __init__(self, input_dim):
        super(single_layer, self).__init__()
        self.a = nn.Linear(input_dim, 1, bias=False)
        # self.a.weight.data.fill_(1)
        # self.a.weight.data[0][1] = 0
        # self.a.weight.data[0][2] = 0
        # self.a.weight.data[0][3] = 0
        # self.a.weight.data[0][4] = 0
        # self.a.weight.data[0][5] = 0

    def forward(self, x):
        return F.relu(self.a.forward(x))

