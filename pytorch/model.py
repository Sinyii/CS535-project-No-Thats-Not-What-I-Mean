import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFNN(nn.Module):
    def __init__(self, hidden_size=3072, class_size=2):
        super(FFNN, self).__init__()
        self.h = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, class_size)
        
    def forward(self, signal):
        h1 = self.h(signal)
        h2 = self.h(h1)
        output = self.out(h2)
        return output

        
