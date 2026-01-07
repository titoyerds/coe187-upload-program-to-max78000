import torch
import numpy as np
from datasets.ripe_vs_unripe import LoafvsNotLoaf

# Path to your dataset root folder
root_dir = "C:/ai8x-training/data"  # adjust to where 'loaf_vs_notloaf/train' and 'loaf_vs_notloaf/test' exist

# Load the test dataset
dataset = LoafvsNotLoaf(root_dir=root_dir, d_type='test')

# Get the first sample
x, _ = dataset[0]

# Add batch dimension for consistency with train.py
x = np.expand_dims(x, axis=0).astype('int64')  # shape becomes (1, C, H, W) if image is (C,H,W)
x = np.clip(x, -128, 127)

# Save as expected filename
np.save('sample_loaf_vs_notloaf.npy', x)
print("Saved sample_loaf_vs_notloaf.npy successfully!")
