import torch.cuda
import torch.nn as nn
from src import  N_CLASSES, FEATURE_DIM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.out = nn.Linear(FEATURE_DIM, N_CLASSES)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x