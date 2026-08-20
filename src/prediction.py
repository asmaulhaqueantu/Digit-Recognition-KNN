import joblib


def predict_digit(model, flattened_image):
    """Predict one or more digits from flattened image data."""
    return model.predict(flattened_image)


def save_model(model, path="models/KNNModel_Job"):
    joblib.dump(model, path)


def load_model(path="models/KNNModel_Job"):
    return joblib.load(path)
