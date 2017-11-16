import torch.nn as nn
import torch.nn.functional as F

class double_layer(nn.Module):
    def __init__(self, input_dim,hidden):
        super(double_layer, self).__init__()
        # self.a = nn.Linear(input_dim, hidden, bias=False)
        # self.b = nn.Linear(hidden, 1, bias=False)
        self.a = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # return F.relu(self.b(F.sigmoid(self.a.forward(x))))
        return self.a.forward(x).alphatronrelu()

