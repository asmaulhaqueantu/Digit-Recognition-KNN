from sklearn.neighbors import KNeighborsClassifier


def create_knn(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def train_knn(X_train_flattened, y_train, n_neighbors=3):
    """Create and train the KNN classifier."""
    knn = create_knn(n_neighbors)
    knn.fit(X_train_flattened, y_train)
    return knn


def evaluate_knn(knn, X_test_flattened, y_test):
    """Return classification accuracy on the test set."""
    return knn.score(X_test_flattened, y_test)
