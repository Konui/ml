import torch
import torch.nn as nn
import numpy as np

class DQN(nn.model):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
    
    def forward(self, x):
        return self.fc(x)