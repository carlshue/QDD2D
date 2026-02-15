"""
File: log_quadtree.py
Project: Adaptive Hierarchical Diffusion

Description:
    Script de prueba y logging completo para MNIST -> Quadtree.
    Mide errores, profundidad, nodos hoja, y permite inspeccionar la construcción.
"""

import torch
from torchvision import datasets, transforms # type: ignore
from structures.quadtree import Quadtree
import numpy as np

# --- Funciones auxiliares ---
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def log_node(node, level=0):
    indent = "  " * level
    if node.is_leaf():
        print(f"{indent}Leaf | x:{node.x} y:{node.y} size:{node.size} value:{node.value:.4f}")
    else:
        print(f"{indent}Node | x:{node.x} y:{node.y} size:{node.size} depth:{node.depth}")
        for child in node.children:
            log_node(child, level+1)

def max_depth_and_leaves(node, current_depth=0):
    if node.is_leaf():
        return current_depth, 1
    else:
        depths = []
        leaves = 0
        for child in node.children:
            d, l = max_depth_and_leaves(child, current_depth +1)
            depths.append(d)
            leaves += l
        return max(depths), leaves

# --- Cargar MNIST ---
transform = transforms.ToTensor()
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
img, label = mnist[0]
img_np = img[0].numpy()

# --- Construir Quadtree ---
qt = Quadtree(img_np, max_depth=None, threshold=0.0)
recon = qt.to_image()

# --- Log completo ---
print("===== QUADTREE LOG =====")
log_node(qt.root)

max_d, num_leaves = max_depth_and_leaves(qt.root)
print("\n===== ESTADÍSTICAS =====")
print(f"Max depth reached: {max_d}")
print(f"Total leaf nodes: {num_leaves}")

# --- Error de reconstrucción ---
error = mse(img_np, recon)
print(f"MSE reconstruction error: {error:.6f}")

# --- Comprobar si hay huecos en la reconstrucción ---
missing = np.sum(np.isnan(recon))
print(f"Pixels missing (NaN): {missing}")
if missing == 0:
    print("No hay huecos detectados.")
else:
    print("Atención: hay huecos en la reconstrucción.")

# --- Visualización opcional ---
try:
    import matplotlib.pyplot as plt # type: ignore
    from utils.visualization import plot_image, plot_quadtree_overlay
    plot_image(img_np, title="Original MNIST")
    plot_image(recon, title="Quadtree Reconstruction")
    plot_quadtree_overlay(img_np, qt, title="Quadtree Overlay")
except ImportError:
    print("Matplotlib no disponible, omitiendo visualización.")
