# ✍️ Handwritten Digit Classification using KNN

> A complete MNIST handwritten-digit classification project built with **Python, TensorFlow/Keras, OpenCV, Pandas, Matplotlib, Seaborn, and Scikit-learn**.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KNN-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5c3ee8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-111111)](https://keras.io/api/datasets/mnist/)

---

## 📌 Project Overview

This project recognizes handwritten digits from **0 to 9** using the **MNIST dataset** and a **K-Nearest Neighbors (KNN)** classifier.

The project was developed step-by-step in a Jupyter Notebook and then organized into a clean, reusable GitHub project structure.

The workflow includes:

- Loading the MNIST handwritten-digit dataset
- Inspecting image shapes and labels
- Visualizing digits using Matplotlib and Seaborn
- Normalizing pixel values from `0–255` to `0–1`
- Flattening each `28 × 28` image into `784` features
- Training a KNN classifier with `k = 3`
- Evaluating the classifier on the MNIST test set
- Testing individual predictions
- Performing an image-rotation experiment with OpenCV
- Reading a handwritten digit from an external image
- Inverting, resizing, normalizing, and flattening the external image
- Predicting the external digit using the trained KNN model
- Saving/loading the trained model with Joblib

---

## 🎯 Objective

The main objective is to understand a complete classical machine-learning workflow for image classification:

```text
Raw Image
   ↓
Preprocessing
   ↓
Feature Transformation
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Evaluation
```

This project is intentionally based on **KNN rather than a neural network**, making it useful for understanding the fundamentals of image preprocessing and traditional supervised learning.

---

## 🧠 Model

The classifier used in this project is:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_flattened, y_train)
```

### Why KNN?

KNN is a simple supervised-learning algorithm that classifies a new sample based on the labels of its nearest training examples.

For this project:

- `k = 3`
- Distance-based classification
- Input: `784` normalized pixel features
- Output: one digit label from `0–9`

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

| Property | Value |
|---|---:|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 |
| Pixel values | 0–255 |
| Normalized values | 0–1 |
| Flattened features | 784 |
| Classes | 10 (0–9) |

The dataset is loaded using:

```python
from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
```

---

## 📈 Result

In the original project run, the KNN classifier achieved:

```text
Test Accuracy: 0.9705
```

### ⭐ Accuracy: **97.05%**

This result was obtained using:

```python
KNeighborsClassifier(n_neighbors=3)
```

> Note: exact runtime results can vary slightly depending on the software environment and implementation details.

---

## 🔬 Project Workflow

```text
                 ┌─────────────────────┐
                 │    MNIST Dataset    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │   Load 60k / 10k    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    Visualization    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Normalize /255      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ 28×28 → 784         │
                 │     Flattening      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │      KNN (k=3)      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │   Test Accuracy     │
                 │      97.05%         │
                 └──────────┬──────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       Rotation Test              External Image
              ↓                           ↓
           OpenCV                  Grayscale
                                      ↓
                                   Invert
                                      ↓
                                  Resize 28×28
                                      ↓
                                  Normalize
                                      ↓
                                  Flatten 784
                                      ↓
                                  KNN Prediction
                                      ↓
                                  Predicted Digit
```

---

## 🖼️ Image Preprocessing

For an external handwritten image, the project follows this pipeline:

```python
modified_image = cv2.imread(
    "modified.png",
    cv2.IMREAD_GRAYSCALE
)

img_resizedM = cv2.bitwise_not(modified_image)

img_resizedM = cv2.resize(
    img_resizedM,
    (28, 28),
    interpolation=cv2.INTER_LINEAR
)

img_resizedM = img_resizedM / 255.0

img_resizedM_Flat = img_resizedM.reshape(1, 28 * 28)

prediction = knn.predict(img_resizedM_Flat)
```

### Pipeline

```text
Original handwritten image
          ↓
      Grayscale
          ↓
     Bitwise NOT
          ↓
     Resize 28×28
          ↓
       / 255
          ↓
      Flatten
          ↓
    784 features
          ↓
       KNN
          ↓
  Predicted digit
```

---

## 🔄 Rotation Experiment

The notebook also experiments with rotating a test digit using OpenCV:

```python
h, w = X_test[6].shape[:2]

center = (w / 2, h / 2)
mat = cv2.getRotationMatrix2D(center, 90, 1)

rotimg = cv2.warpAffine(
    X_test[6],
    mat,
    (w, h)
)
```

The rotated image is then flattened and passed to the same KNN model.

This experiment demonstrates an important practical issue in image classification:

> **A machine-learning model can be sensitive to changes in image orientation and preprocessing.**

---

## 🧪 Individual Prediction

The project also tests individual MNIST samples:

```python
knn.predict(X_test_flattened[[0]])
```

Example result:

```text
array([7], dtype=uint8)
```

The predicted value can be compared with:

```python
y_test[0]
```

---

## 💾 Model Saving

The trained KNN model can be saved using Joblib:

```python
import joblib as jbl

jbl.dump(knn, "models/KNNModel_Job")
```

It can later be loaded without retraining:

```python
model = joblib.load("models/KNNModel_Job")
```

### ⚠️ GitHub note

A KNN model trained on all 60,000 MNIST samples can become very large because KNN retains the training data.

For that reason, the generated model file is **not required to be committed to the repository**. If you want to version it, use **Git LFS** or an external model-storage service.

---

## 🗂️ Project Structure

```text
Handwritten-Digit-Classification-KNN/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
│
├── train.py
├── predict_external.py
│
├── notebooks/
│   └── handwritten_digit_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── image_processing.py
│   └── prediction.py
│
├── models/
│   └── .gitkeep
│
├── images/
│   ├── .gitkeep
│   └── README.md
│
└── results/
    ├── .gitkeep
    └── README.md
```

---

## 🛠️ Technologies & Libraries

- **Python**
- **TensorFlow / Keras** — MNIST dataset
- **NumPy** — numerical operations
- **Pandas** — tabular inspection
- **Matplotlib** — image visualization
- **Seaborn** — heatmap visualization
- **Scikit-learn** — KNN classifier
- **OpenCV** — image processing and rotation
- **Joblib** — model persistence
- **Jupyter Notebook** — experimentation and documentation

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Handwritten-Digit-Classification-KNN.git
cd Handwritten-Digit-Classification-KNN
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Training Pipeline

```bash
python train.py
```

The script:

1. Downloads/loads MNIST
2. Normalizes the images
3. Flattens images into 784 features
4. Trains KNN with `k=3`
5. Evaluates the model
6. Saves the trained model

Expected output includes:

```text
Training data shape: (60000, 28, 28)
Test data shape: (10000, 28, 28)
Flattened training shape: (60000, 784)
KNN test accuracy: 0.9705
Model saved to models/KNNModel_Job
```

---

## ✍️ Predict Your Own Handwritten Digit

Place an image inside:

```text
images/modified.png
```

Then run:

```bash
python predict_external.py
```

Or specify another image:

```bash
python predict_external.py path/to/your/image.png
```

The script displays the processed image and prints:

```text
Predicted digit: X
```

---

## 📓 Jupyter Notebook

The complete cleaned notebook is available here:

```text
notebooks/handwritten_digit_classification.ipynb
```

The notebook preserves the original experimental workflow, including:

- Dataset loading
- Data inspection
- Heatmap
- Image visualization
- Normalization
- Flattening
- KNN training
- Accuracy evaluation
- Individual predictions
- Pandas inspection of 784 features
- OpenCV rotation
- External image processing
- External prediction
- Joblib model saving

---

## 🧩 Source Code Organization

### `src/data_loader.py`

Loads the MNIST dataset.

### `src/preprocessing.py`

Contains normalization and image-flattening utilities.

### `src/model.py`

Creates, trains, and evaluates the KNN classifier.

### `src/image_processing.py`

Handles external image loading, inversion, resizing, normalization, and rotation.

### `src/prediction.py`

Handles prediction and Joblib model persistence.

### `train.py`

Runs the complete training and evaluation pipeline.

### `predict_external.py`

Runs inference on an external handwritten image.

---

## 📌 Limitations

This project is intentionally simple and has several limitations:

- KNN can be computationally expensive for large datasets.
- Prediction can be slower than many trained neural-network models.
- External images may require careful preprocessing.
- Rotation, scale, thickness, background, and alignment can affect predictions.
- The external image should resemble the MNIST style for better results.
- The full KNN model can be large because training samples are retained.

---

## 🔮 Future Improvements

Possible improvements include:

- Compare different values of `k`
- Add a confusion matrix
- Generate a classification report
- Compare KNN with Logistic Regression
- Compare KNN with SVM
- Build a CNN using TensorFlow/Keras
- Add automatic image centering
- Add thresholding and noise removal
- Add a simple Streamlit web interface
- Create an interactive digit-drawing canvas
- Add automated model evaluation plots
- Track experiments and metrics

---

## 📚 Learning Outcomes

Through this project, the following concepts were practiced:

- Dataset loading
- Exploratory data visualization
- Image normalization
- Feature engineering
- Reshaping and flattening
- Supervised machine learning
- KNN classification
- Model evaluation
- Image preprocessing with OpenCV
- Model persistence with Joblib
- Organizing a machine-learning project for GitHub

---

## 👤 Author

**Handwritten Digit Classification — KNN**

This project was developed as a machine-learning/data-analysis portfolio project based on the MNIST handwritten digit dataset.

---

## ⭐ If you find this project useful

Feel free to fork the repository, experiment with different classifiers, improve the preprocessing pipeline, and build your own version of the application.

