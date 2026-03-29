#!/usr/bin/env python3
"""Create rearing-pose initialization for Exp B."""
import pickle, torch, numpy as np, os, copy

with open('results/fitting/production_keyframes_part3/params/step_2_frame_010080.pkl', 'rb') as f:
    base = pickle.load(f)

# Clone all params to new dict
rearing = {}
for k, v in base.items():
    if isinstance(v, torch.Tensor):
        rearing[k] = v.detach().clone()
    else:
        rearing[k] = copy.deepcopy(v)

rot = rearing['rotation'].cpu().numpy().flatten()
print(f'Current rotation: {rot}')

# Modify rotation: add ~60 deg pitch (Z-axis)
with torch.no_grad():
    rearing['rotation'][0, 2] += 1.05
    rearing['trans'][0, 1] -= 20.0

print(f'Rearing rotation: {rearing["rotation"].cpu().numpy().flatten()}')
print(f'Rearing trans: {rearing["trans"].cpu().numpy().flatten()}')

# Set requires_grad AFTER modification
for k, v in rearing.items():
    if isinstance(v, torch.Tensor):
        rearing[k] = v.requires_grad_(True)

out_dir = 'results/fitting/rearing_test_exp_b'
os.makedirs(f'{out_dir}/params', exist_ok=True)
os.makedirs(f'{out_dir}/obj', exist_ok=True)
with open(f'{out_dir}/params/step_2_frame_010079.pkl', 'wb') as f:
    pickle.dump(rearing, f)
print('Saved rearing init')
