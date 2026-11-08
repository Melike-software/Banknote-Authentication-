import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ağırlıkları başlangıçta sıfır yap
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent
        for epoch in range(self.epochs):

            # Tahmin hesapla
            y_pred = np.dot(X, self.weights) + self.bias

            # Hata hesapla
            error = y_pred - y

            # MSE kayıp fonksiyonu
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # Gradyanları hesapla
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            # Ağırlıkları güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_values(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        predictions = self.predict_values(X)
        return (predictions >= 0.5).astype(int)
