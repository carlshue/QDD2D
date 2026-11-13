"""Punto único para cargar un conjunto de imágenes binarias por nombre."""
DATASETS = ("mnist", "quickdraw")


def load_images(name="mnist", train=True):
    """Devuelve (imgs (N, S, S) uint8 en {0,1}, labels (N,) int64). S es potencia de 2."""
    if name == "mnist":
        from data.binary_mnist import load_binary_mnist
        return load_binary_mnist(train=train)
    if name == "quickdraw":
        from data.quickdraw import load_quickdraw
        return load_quickdraw(train=train)
    raise ValueError(f"conjunto de datos desconocido: {name}")


def ckpt_dataset(path):
    """Con qué datos se entrenó un checkpoint (los antiguos no lo guardan: MNIST)."""
    import torch
    return torch.load(path, map_location="cpu")["args"].get("dataset", "mnist")
