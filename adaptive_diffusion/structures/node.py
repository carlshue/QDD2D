"""
File: node.py
Project: Adaptive Hierarchical Diffusion

Description:
    Defines the QuadtreeNode class representing a single node
    in a hierarchical spatial partition.
"""
"""
File: node.py
Project: Adaptive Hierarchical Diffusion

Description:
    Defines the QuadtreeNode class representing a single node
    in a hierarchical spatial partition.
"""
class QuadtreeNode:
    def __init__(self, x, y, size, value=None, depth=0):
        self.x = x
        self.y = y
        self.size = size
        self.value = value
        self.children = []
        self.depth = depth

    def is_leaf(self):
        return len(self.children) == 0

    def subdivide(self, values, max_depth=None, threshold=0.0):
        import numpy as np

        block = values[self.y:self.y+self.size, self.x:self.x+self.size]
        var = np.var(block)

        # Condición de parada: homogéneo, tamaño 1 o profundidad máxima
        if self.size <= 1 or var <= threshold or (max_depth is not None and self.depth >= max_depth):
            self.value = np.mean(block)
            return

        # Subdividir en 4 cuadrantes exactos
        half = self.size // 2
        remain = self.size - half
        quadrants = [
            (0, 0, half, half),
            (half, 0, remain, half),
            (0, half, half, remain),
            (half, half, remain, remain)
        ]

        self.children = []
        for dx, dy, w, h in quadrants:
            child = QuadtreeNode(self.x + dx, self.y + dy, max(w,h), depth=self.depth+1)
            child.subdivide(values, max_depth=max_depth, threshold=threshold)
            self.children.append(child)
