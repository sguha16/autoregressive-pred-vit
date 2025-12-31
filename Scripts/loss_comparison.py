# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 17:39:33 2025

@author: uig67136
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the 3 loss arrays
loss_1 = np.load('1target/losses_1targets.npy')
loss_2 = np.load('2targets/losses_2targets.npy')
loss_3 = np.load('3targets/losses_3targets.npy')  

# Plot
plt.figure(figsize=(10, 6))
plt.plot(loss_1, label='1 target', linewidth=2)
plt.plot(loss_2, label='2 targets', linewidth=2)
plt.plot(loss_3, label='3 targets', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Training Loss vs Scene Complexity', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=150)
plt.show()