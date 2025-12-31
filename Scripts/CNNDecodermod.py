# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 13:46:50 2025

@author: uig67136
"""
import torch
import torch.nn as nn
class CNNDecodermod(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        
        self.fc = nn.Linear(embed_dim, 64 * 8 * 8)
        
        self.decoder = nn.Sequential(
            # 8×8 → 16×16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 16×16 → 32×32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 32×32 → 64×64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)
        x = self.decoder(x)
        return x