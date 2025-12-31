# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 07:06:54 2025

@author: uig67136
"""

import torch
from torch.utils.data import Dataset, DataLoader

class RadarSequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        sequences: (200, 5, 1, 64, 64)
        """
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (5, 1, 64, 64)
        
        # Input: first 4 frames (t=0,1,2,3)
        input_seq = seq[:4]  # (4, 1, 64, 64)
        
        # Target: 5th frame (t=4)
        target = seq[4]  # (1, 64, 64)
        
        return torch.tensor(input_seq, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.float32)

#Dataloader
