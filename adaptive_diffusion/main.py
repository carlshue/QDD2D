import torch
from data.mnist_loader import get_mnist_dataloaders
from structures.quadtree import Quadtree
from utils.visualization import plot_three_images
import numpy as np

# --- Funciones de logging ---
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
            d, l = max_depth_and_leaves(child, current_depth + 1)
            depths.append(d)
            leaves += l
        return max(depths), leaves

# --- Cargar MNIST ---
train_loader, _ = get_mnist_dataloaders(batch_size=1)

for img, label in train_loader:
    img_np = img[0,0].numpy()

    # --- Construir Quadtree ---
    qt = Quadtree(img_np, max_depth=None, threshold=0.0)  # subdividir hasta pixel si hace falta
    recon_bin = qt.to_binary_image(img_np)  # reconstrucción binaria directamente

    # --- Visualización de las tres imágenes ---
    plot_three_images(img_np, recon_bin, qt)

    # --- Logging completo ---
    print("===== QUADTREE LOG =====")
    log_node(qt.root)

    max_d, num_leaves = max_depth_and_leaves(qt.root)
    print("\n===== ESTADÍSTICAS =====")
    print(f"Max depth reached: {max_d}")
    print(f"Total leaf nodes: {num_leaves}")

    error = mse(img_np, recon_bin)
    print(f"MSE reconstruction error: {error:.6f}")

    missing = np.sum(np.isnan(recon_bin))
    print(f"Pixels missing (NaN): {missing}")
    if missing == 0:
        print("No hay huecos detectados.")
    else:
        print("Atención: hay huecos en la reconstrucción.")

    break  # solo el primer batch
