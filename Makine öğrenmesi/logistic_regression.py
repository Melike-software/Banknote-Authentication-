import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        # Sayısal taşmaları önlemeye
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Başlangıç ağırlıkları
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):

            # Doğrusal birleşim
            z = np.dot(X, self.weights) + self.bias

            # Sigmoid ile olasılık hesapı
            y_pred = self.sigmoid(z)

            # Binary Cross Entropy kaybı
            epsilon = 1e-15
            y_safe = np.clip(y_pred, epsilon, 1 - epsilon)

            loss = -np.mean(
                y * np.log(y_safe)
                + (1 - y) * np.log(1 - y_safe)
            )

            self.loss_history.append(loss)

            # Gradyanları hesapla
            error = y_pred - y

            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error) / n_samples

            # Ağırlıkları güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
