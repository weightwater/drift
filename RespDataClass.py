import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class RespData(Dataset):

    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return self.x[idx]
    
    def __len__(self):
        return len(self.x)