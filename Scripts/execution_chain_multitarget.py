# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 05:02:24 2025

@author: uig67136
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:37:51 2025

@author: uig67136
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

#from SynthData import generate_dataset
#from SynthData import generate_single_rd_map
from dataloader_sequence import RadarSequenceDataset
from VIT_RadarSequencePred_model import RadarSequencePredictor
from KalmanF import KalmanFilterCV
from SynthData_multi import generate_single_rd_map
from SynthData_multi import generate_dataset

# =======================
# 1. Generate Synthetic Data
# =======================
N_samples_train=500
N_samples_test=100
num_targets=1
print("Step 1: Generating synthetic dataset...")
#X, R_hist_all, v_hist_all, target_vel_arr, Y = generate_dataset(N_samples)
#gen dataset new--for multi target
X_train, R_hist_all_train, v_hist_all_train, target_accel_train, Y_train = generate_dataset(N_samples_train, num_targets)
X_test, R_hist_all_test, v_hist_all_test, target_accel_test, Y_test = generate_dataset(N_samples_test, num_targets)

print("Synthetic train and test dataset generated.")
print("X_train shape RD maps list:", X_train.shape)  # (200, 5, 1, 64, 64)

# Visualize RD maps from t=0:4 for 1 sample
plt.figure(figsize=(20, 4))  # Make it wide enough for 5 subplots
for idx in range(5):
    plt.subplot(1, 5, idx+1)  # ← FIXED: (1 row, 5 cols, position idx+1)
    plt.imshow(X_train[10, idx, 0, :, :], aspect='auto')
    plt.xlabel("Doppler")
    plt.ylabel("Range")
    plt.colorbar()
    plt.title(f"t={idx}")  # ← Show time index
plt.tight_layout()
plt.show()

#Inputs available from synthetic data:
#1)X-5 sequential RD maps per sample 
#2)R_hist_all, a list of cells, each cell has 5 ranges, 1 per time instance
#3)V_hist_all, a list of cells, each cell has 5 velocities, 1 per time instance
#4) target_vel_arr-fixed/constant vel of every target/sample [0,1,2,3,4]
#5)Y=zeroes for RD map of 5th time instance of every sample--which we are trying to predict [N_samples,1,64,64]
#=========================
#Kalman filter baseline-->only for single target case
#=========================
R_pred_store=[]
v_pred_store=[]
RD_pred_store=[]
mse_store=[]
for i in range(N_samples_test):
    R_hist = R_hist_all_test[i][0]   # [R0, R1, R2, R3, R4]
    v_hist = v_hist_all_test[i][0]   # [v0, v1, v2, v3, v4]
    
    kf = KalmanFilterCV(dt=1.0)
    kf.init_state(R_hist[0], v_hist[0],a0=0.0)
    for t in range(1, 4):
        kf.predict()  # predict state at time t
        z = np.array([R_hist[t], v_hist[t]])
        kf.update(z)  # correct using measurement
    x_pred = kf.predict()
    R_pred, v_pred,acc_pred = x_pred.flatten()
    #error per sample
    error_R = R_pred - R_hist[4]#5th time instance
    error_v = v_pred - v_hist[4]#5th time instance

    print("err R",error_R)
    print("err v",error_v)

    tgt_pred = {
        'R': R_pred,
        'v': v_pred,
        'amp': 100  #default
        }
    RD_pred = generate_single_rd_map([tgt_pred],noise_std=0.0)
    mse_i = np.mean((RD_pred - X_test[i,4,0,:,:].squeeze()) ** 2)

    #storing results from KF--range pred, vel pred and RD maps generated
    R_pred_store.append(R_pred)
    v_pred_store.append(v_pred)
    RD_pred_store.append(RD_pred)
    mse_store.append(mse_i)
#plot some results---eg for sample 1 

print("\nStep KF prediction completed...")
plt.figure()
plt.imshow(RD_pred_store[2].squeeze(),aspect='auto')
plt.xlabel("Doppler-pred")
plt.ylabel("Range")
plt.colorbar()

plt.figure()
plt.imshow(X_test[2,4,0,:,:],aspect='auto')
plt.xlabel("Doppler-GT")
plt.ylabel("Range")
plt.colorbar()
print('KF Loss (MSE)',np.mean(mse_store))

# =======================
# 2. Create DataLoader
# =======================
print("\nStep 2: Creating dataloader...")
train_dataset = RadarSequenceDataset(X_train)
test_dataset = RadarSequenceDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test dataloader
for input_seq, target in train_loader:
    print(f"Input sequence shape: {input_seq.shape}")   # (32, 4, 1, 64, 64)
    print(f"Target shape: {target.shape}")              # (32, 1, 64, 64)
    break

# =======================
# 3. Create Model
# =======================
print("\nStep 3: Creating model...")
model = RadarSequencePredictor(
    seq_len=4,
    embed_dim=192,
    num_heads=3,
    num_layers=6
)
print("Model created.")


# Test forward pass
print("\nTesting forward pass...")
for input_seq, target in train_loader:
    pred = model(input_seq)
    print(f"Input: {input_seq.shape}")    # (32, 4, 1, 64, 64)
    print(f"Prediction: {pred.shape}")    # (32, 1, 64, 64)
    print(f"Target: {target.shape}")      # (32, 1, 64, 64)
    break

# =======================
# 4. Training
# =======================
print("\nStep 4: Training...")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()  # Mean squared error for prediction
optimizer = optim.Adam(model.parameters(), lr=1e-4)

fixed_sample_input, fixed_sample_target = next(iter(test_loader))
fixed_sample_input = fixed_sample_input.to(device)
fixed_sample_target = fixed_sample_target.to(device)

num_epochs = 50
loss_list=[]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for input_seq, target in train_loader:
        input_seq = input_seq.to(device)
        target = target.to(device)
        
        # Forward
        pred = model(input_seq)
        loss = criterion(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    # ← Add visualization every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            
            pred = model(fixed_sample_input)
            
            # Plot
            plt.figure(figsize=(12, 3))
            plt.subplot(131)
            plt.imshow(fixed_sample_input[0, -1, 0].cpu(), aspect='auto')
            plt.title('t=3')
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(pred[0, 0].cpu(), aspect='auto')
            plt.title(f'Pred Epoch {epoch+1}')
            plt.colorbar()
            
            plt.subplot(133)
            plt.imshow(fixed_sample_target[0, 0].cpu(), aspect='auto')
            plt.title('Truth')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f'pred_ep{epoch+1}.png')
            plt.close()

print("Training complete!")
np.save(f'losses_{num_targets}targets.npy', loss_list)

# After training completes
# model.eval()
# with torch.no_grad():
#     sample_input, sample_target = next(iter(train_loader))
#     sample_input = sample_input.to(device)
#     sample_pred = model(sample_input)
# After training completes
model.eval()
test_loss = 0
with torch.no_grad():
    for input_seq, target in test_loader:
        input_seq = input_seq.to(device)
        target = target.to(device)
        pred = model(input_seq)
        loss = criterion(pred, target)
        test_loss += loss.item()

vit_test_mse = test_loss / len(test_loader)
print(f"VIT Test MSE: {vit_test_mse:.6f}")

# Get one sample for visualization
with torch.no_grad():
    sample_input, sample_target = next(iter(test_loader))
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)
    sample_pred = model(sample_input)

# Now visualize
import matplotlib.pyplot as plt

idx = 0  # First sample in batch
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(sample_input[idx, -1, 0].cpu(), aspect='auto')
plt.title("Last input frame (t=3)")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.imshow(sample_pred[idx, 0].cpu(), aspect='auto')
plt.title("Predicted VIT(t=4)")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(RD_pred_store[idx].squeeze(), aspect='auto')
plt.title("Predicted KF (t=4)")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(sample_target[idx, 0].cpu(), aspect='auto')
plt.title("Ground truth (t=4)")
plt.colorbar()

plt.tight_layout()
plt.show()
# Save model
torch.save(model.state_dict(), 'sequence_predictor.pth')
print("Model saved.")

print("\n✓ All components working!")
