from tensorflow.keras import datasets


def load_mnist():
    """Load the MNIST handwritten digit dataset."""
    return datasets.mnist.load_data()
