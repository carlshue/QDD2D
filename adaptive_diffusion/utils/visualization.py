"""
File: visualization.py
Project: Adaptive Hierarchical Diffusion

Description:
    Utilities to visualize MNIST images and overlay quadtree subdivisions.
"""
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore

def plot_image(img, title="Image"):
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_three_images(original, recon_bin, quadtree, title="MNIST Quadtree Visualization"):
    """
    Plotea:
        1. Imagen original
        2. Reconstrucción binaria
        3. Overlay del quadtree sobre reconstrucción
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.05)  # hueco entre las imágenes

    # 1️⃣ Imagen original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original MNIST")
    axes[0].axis('off')

    # 2️⃣ Reconstrucción binaria
    axes[1].imshow(recon_bin, cmap='gray')
    axes[1].set_title("Binary Quadtree Recon")
    axes[1].axis('off')

    # 3️⃣ Overlay del quadtree
    axes[2].imshow(recon_bin, cmap='gray')  # sobre reconstrucción
    axes[2].set_title("Quadtree Overlay")
    axes[2].axis('off')

    # Dibujar los nodos del quadtree
    def draw_node(node):
        if node.is_leaf():
            color = 'white' if node.value >= 0 else 'black'
            rect = patches.Rectangle(
                (node.x, node.y), node.size, node.size,
                linewidth=1, edgecolor='red', facecolor=color, alpha=0.3
            )
            axes[2].add_patch(rect)
        else:
            for child in node.children:
                draw_node(child)

    draw_node(quadtree.root)
    plt.show()