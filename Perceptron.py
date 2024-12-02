import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = None
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.activation_func = self.unit_step_func
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_hat = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_hat)
                self.weights += update * x_i
                self.bias += update

    def unit_step_func(x):
        return np.where(x > 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_hat = self.activation_func(linear_output)
        return y_hat