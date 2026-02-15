"""
File: quadtree.py
Project: Adaptive Hierarchical Diffusion

Description:
    Implements the Quadtree structure, including subdivision,
    traversal, serialization and image reconstruction.
"""

"""
File: quadtree.py
Project: Adaptive Hierarchical Diffusion

Description:
    Implements the Quadtree structure, including subdivision,
    traversal, serialization and image reconstruction.
"""
import numpy as np
from structures.node import QuadtreeNode

class Quadtree:
    def __init__(self, image, max_depth=None, threshold=0.0):
        """
        image: numpy array 2D
        max_depth: None = subdividir hasta 1 pixel
        threshold: varianza mínima para subdividir
        """
        import numpy as np
        self.image = image
        self.size = image.shape[0]
        self.max_depth = max_depth
        self.threshold = threshold
        self.root = QuadtreeNode(0, 0, self.size)
        self.root.subdivide(image, max_depth=max_depth, threshold=threshold)

    def to_image(self):
        import numpy as np
        img = np.zeros((self.size, self.size))
        self._fill_node(self.root, img)
        return img

    def _fill_node(self, node, img):
        if node.is_leaf():
            img[node.y:node.y+node.size, node.x:node.x+node.size] = node.value
        else:
            for child in node.children:
                self._fill_node(child, img)

    def to_binary_image(self, original_image):
        """
        Reconstruye la imagen a partir del quadtree usando solo blanco o negro.
        Cada nodo hoja se umbraliza según el valor que minimice el error con la subimagen original.
        """
        import numpy as np
        img = np.zeros((self.size, self.size))

        def fill_node_binary(node):
            if node.is_leaf():
                block_orig = original_image[node.y:node.y+node.size, node.x:node.x+node.size]
                val_black = -1.0
                val_white = 1.0
                err_black = np.sum((block_orig - val_black)**2)
                err_white = np.sum((block_orig - val_white)**2)
                val = val_white if err_white < err_black else val_black
                img[node.y:node.y+node.size, node.x:node.x+node.size] = val
            else:
                for child in node.children:
                    fill_node_binary(child)

        fill_node_binary(self.root)
        return img