import numpy as np


def normalize_images(X_train, X_test):
    """Scale pixel values from [0, 255] to [0, 1]."""
    return X_train.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0


def flatten_images(X):
    """Convert N x 28 x 28 images into N x 784 feature vectors."""
    return X.reshape(len(X), 28 * 28)
