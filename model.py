import torch.cuda
import torch.nn as nn
from src import  N_CLASSES, FEATURE_DIM

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(FEATURE_DIM, int(FEATURE_DIM / 2)) 
        self.layer_2 = nn.Linear(int(FEATURE_DIM / 2), int(FEATURE_DIM / 4))
        self.layer_out = nn.Linear(int(FEATURE_DIM / 4), N_CLASSES) 

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(int(FEATURE_DIM / 2))
        self.batchnorm2 = nn.BatchNorm1d(int(FEATURE_DIM / 4))

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x