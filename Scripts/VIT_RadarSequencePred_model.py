# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 07:42:06 2025

@author: uig67136
"""

import torch
import torch.nn as nn
import timm
#from CNNDecoder import CNNDecoder
from CNNDecodermod import CNNDecodermod

class RadarSequencePredictor(nn.Module):
    def __init__(self, seq_len=4, embed_dim=192, num_heads=3, num_layers=6):
        super().__init__()
        
        # ViT encoder for each frame
        self.frame_encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=0,
            img_size=64,
            in_chans=1,
            global_pool='avg'
        )
        
        # Temporal positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Decoder to RD map MLP VIT
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64),
            nn.Sigmoid()
        )
        
        
        #Decoder to RD map CNN
        #self.decoder = CNNDecodermod(embed_dim=192)

        
    def forward(self, x):
        # x: (batch, 4, 1, 64, 64)
        batch_size, seq_len, _, H, W = x.shape
        
        # Encode each frame
        frame_features = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch, 1, 64, 64)
            feat = self.frame_encoder(frame)  # (batch, 192)
            frame_features.append(feat)
        
        # Stack and add positional encoding
        seq_features = torch.stack(frame_features, dim=1)  # (batch, 4, 192)
        seq_features = seq_features + self.pos_encoding
        
        # Transformer
        seq_features = seq_features.transpose(0, 1)  # (4, batch, 192)
        decoded = self.transformer_decoder(seq_features, seq_features)
        last_features = decoded[-1]  # (batch, 192)
        
        #Predict VIT
        predicted = self.decoder(last_features)  # (batch, 4096)
        predicted = predicted.view(batch_size, 1, H, W)
        
        #Predict CNN
        #predicted = self.decoder(last_features)
        
        return predicted

