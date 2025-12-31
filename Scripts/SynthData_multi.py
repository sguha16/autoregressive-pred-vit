# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 15:18:45 2025

@author: uig67136
"""

import numpy as np
from skimage.transform import resize

def generate_single_rd_map(targets,  # ← Changed: now accepts LIST of targets
                           fc=77e9, B=150e6, T_chirp=50e-6,
                           N_samples=64, N_chirps=64, noise_std=0.01):
    """
    Generate a single synthetic Range-Doppler map for multiple targets.
    
    Args:
        targets: list of dicts [{'R':..., 'v':..., 'amp':...}, ...]
    Returns:
        RD_map_cnn: np.array of shape (1, 64, 64)
    """
    c = 3e8  # speed of light
    
    # Time axes
    t_fast = np.linspace(0, T_chirp, N_samples)
    t_slow = np.linspace(0, N_chirps*T_chirp, N_chirps)
    
    # Initialize beat signal
    beat_signal = np.zeros((N_samples, N_chirps), dtype=complex)
    
    # ← CHANGED: Loop through all targets
    for tgt in targets:
        R, v, amp = tgt['R'], tgt['v'], tgt['amp']
        #adding amp attenuation with increasing Range-->
        # Skip targets outside valid range
        if R > 60 or R < 2:
            continue
        
        # Range-dependent amplitude attenuation
        amp_attenuated = amp / (R**2 + 1e-6)
        #----------------------------------------------
        f_range = 2*B*R/(c*T_chirp)
        f_doppler = 2*v*fc/c
        
        # Superposition: add each target's contribution
        beat_signal += amp_attenuated * np.exp(1j*2*np.pi*(
            f_range*t_fast[:,None] + f_doppler*t_slow[None,:]))
    
    # Add noise
    noise = (np.random.normal(0, noise_std, beat_signal.shape) +
             1j*np.random.normal(0, noise_std, beat_signal.shape))
    beat_signal += noise
    
    # 2D FFT
    RD_range = np.fft.fft(beat_signal, axis=0)
    RD_map = np.fft.fft(RD_range, axis=1)
    RD_map_mag = np.abs(RD_map)
    RD_map_mag /= np.max(RD_map_mag)
    
    RD_map_cnn = RD_map_mag[np.newaxis, :, :]  # shape (1, 64, 64)
    
    return RD_map_cnn


def generate_dataset(N_samples=100, num_targets=2):  # ← NEW parameter
    """
    Generate synthetic RD map sequences with multiple targets.
    
    Args:
        N_samples: number of sequences
        num_targets: number of targets per scene (1, 2, or 4)
    
    Returns:
        X: (N_samples, 5, 1, 64, 64) - 5 frames per sequence
        R_hist_all: list of range histories for all targets
        v_hist_all: list of velocity histories for all targets
        target_velocity_arr: velocity profile used
        Y: placeholder
    """
    rdmaps_list = []
    R_hist_all = []
    v_hist_all = []
    
    #targetarget_velocity_arr = [ 1, 2, 3, 4, 5]  # m/s
    target_accel = 1.0  # m/s² (constant acceleration)
    target_time_arr = [0, 1, 2, 3, 4]  # sec
    
    for i in range(N_samples):
        
        # ← NEW: Initialize multiple targets
        targets = []
        for tgt_idx in range(num_targets):
            tgt = {
                'R': np.random.uniform(10, 30),  # Random initial range
                'v': np.random.uniform(2, 4),  # Start at 0 m/s
                'amp': 100  # initial amplitude
            }
            targets.append(tgt)
        
        rdmaps_list_persample = []
        
        # Store history for each target separately
        R_hist_sample = [[] for _ in range(num_targets)]
        v_hist_sample = [[] for _ in range(num_targets)]
        
        # Initialize history at t=0
        for tgt_idx, tgt in enumerate(targets):
            R_hist_sample[tgt_idx].append(tgt['R'])
            v_hist_sample[tgt_idx].append(tgt['v'])
        
        # ← CHANGED: Generate sequence
        for time_idx in range(len(target_time_arr)):
            
            # Update each target's position and velocity
            if time_idx > 0:
                dt = target_time_arr[time_idx] - target_time_arr[time_idx - 1]
                
                for tgt_idx, tgt in enumerate(targets):
                    tgt['v'] = tgt['v']+target_accel*dt
                    tgt['R'] += tgt['v'] * dt
                    
                    # Store history
                    R_hist_sample[tgt_idx].append(tgt['R'])
                    v_hist_sample[tgt_idx].append(tgt['v'])
            
            # Generate RD map with ALL targets
            rd_map = generate_single_rd_map(targets)  # ← Pass entire list
            rdmaps_list_persample.append(rd_map)
        
        rdmaps_list.append(rdmaps_list_persample)
        R_hist_all.append(R_hist_sample)  # List of lists
        v_hist_all.append(v_hist_sample)
    
    X = np.stack(rdmaps_list, axis=0)  # (N_samples, 5, 1, 64, 64)
    Y = np.zeros((N_samples, 1, 1, 64, 64))  # Placeholder
    
    return X, R_hist_all, v_hist_all, target_accel, Y