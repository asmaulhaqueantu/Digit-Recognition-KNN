import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.image_processing import load_external_image, prepare_external_image
from src.prediction import load_model, predict_digit


def main():
    parser = argparse.ArgumentParser(
        description="Predict a handwritten digit from an external image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="images/modified.png",
        help="Path to the handwritten digit image."
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    model = load_model("models/KNNModel_Job")
    original = load_external_image(image_path)
    processed, flattened = prepare_external_image(original)

    prediction = predict_digit(model, flattened)[0]

    print("Predicted digit:", prediction)

    plt.imshow(processed, cmap="gray")
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
