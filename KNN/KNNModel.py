import numpy as np


class KNNModel:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = np.linalg.norm(self.X_train - sample, axis=1)
            indices = np.argsort(distances)[:self.n_neighbors]
            neighbors_labels = self.y_train[indices]
            unique_labels, counts = np.unique(neighbors_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)