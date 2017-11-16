import torch.nn as nn
import torch.nn.functional as F

class three_layer(nn.Module):
    def __init__(self, input_dim,hidden):
        super(three_layer, self).__init__()
        self.a = nn.Linear(input_dim, hidden, bias=False)
        self.b = nn.Linear(hidden, 1, bias=False)
        self.a.weight.data.uniform_()
        self.b.weight.data.uniform_()

    def forward(self, x):
        return F.relu(self.b(F.sigmoid(self.a.forward(x))))

