import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class FCAE(nn.Module):

    def __init__(self, inputDim):
        super(FCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, inputDim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
