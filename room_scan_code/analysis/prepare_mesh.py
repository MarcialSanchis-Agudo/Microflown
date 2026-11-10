#!/usr/bin/env python3
"""prepare_mesh.py — vertices of the photogrammetric room model (data/textured_output.obj) and their colours
sampled from its texture (data/textured_output.jpg) at each vertex's texture coordinate.  Writes
results/room_mesh_vertices.npy and results/room_mesh_colors.npy (Fig. 1 and the registration check only)."""
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
V, VT = [], []
with open('data/textured_output.obj') as f:
    for line in f:
        if line.startswith('v '): V.append(line.split()[1:4])
        elif line.startswith('vt '): VT.append(line.split()[1:3])
V = np.array(V, np.float32); VT = np.array(VT, float)
im = np.asarray(Image.open('data/textured_output.jpg').convert('RGB')); H, W = im.shape[:2]
x = np.clip((VT[:, 0] * (W - 1)).round().astype(int), 0, W - 1); y = np.clip(((1 - VT[:, 1]) * (H - 1)).round().astype(int), 0, H - 1)
np.save('results/room_mesh_vertices.npy', V); np.save('results/room_mesh_colors.npy', im[y, x][:len(V)])
print(f'{len(V)} vertices, {len(VT)} texture coordinates')
