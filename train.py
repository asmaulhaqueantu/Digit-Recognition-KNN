from pathlib import Path

from src.data_loader import load_mnist
from src.preprocessing import normalize_images, flatten_images
from src.model import train_knn, evaluate_knn
from src.prediction import save_model


def main():
    # 1. Load MNIST
    (X_train, y_train), (X_test, y_test) = load_mnist()

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)

    # 2. Normalize pixel values
    X_train, X_test = normalize_images(X_train, X_test)

    # 3. Flatten 28x28 images into 784 features
    X_train_flattened = flatten_images(X_train)
    X_test_flattened = flatten_images(X_test)

    print("Flattened training shape:", X_train_flattened.shape)

    # 4. Train KNN
    knn = train_knn(X_train_flattened, y_train, n_neighbors=3)

    # 5. Evaluate
    accuracy = evaluate_knn(knn, X_test_flattened, y_test)
    print(f"KNN test accuracy: {accuracy:.4f}")

    # 6. Save model
    Path("models").mkdir(exist_ok=True)
    save_model(knn, "models/KNNModel_Job")
    print("Model saved to models/KNNModel_Job")


if __name__ == "__main__":
    main()
