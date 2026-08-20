import cv2
import numpy as np


def rotate_digit(image, angle=90):
    """Rotate a square digit image around its center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))


def load_external_image(path):
    """Read an external handwritten image as grayscale."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def prepare_external_image(image, invert=True, size=(28, 28)):
    """
    Prepare an external handwritten digit for MNIST/KNN:
    grayscale -> optional inversion -> resize -> normalize -> flatten.
    """
    processed = image.copy()

    if invert:
        processed = cv2.bitwise_not(processed)

    processed = cv2.resize(processed, size, interpolation=cv2.INTER_LINEAR)
    processed = processed.astype(np.float32) / 255.0
    flattened = processed.reshape(1, 28 * 28)

    return processed, flattened
