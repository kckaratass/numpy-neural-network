import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.1,
        seed=42
    ):
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)

        # Initialize weights
        self.W1 = rng.normal(
            0, 0.1,
            size=(input_size, hidden_size)
        )

        self.W2 = rng.normal(
            0, 0.1,
            size=(hidden_size, output_size)
        )

        # Initialize biases
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

        # Store training loss
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    def forward(self, X):

        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y):

        m = X.shape[0]

        # Output layer gradients
        dZ2 = (
            (2 / m)
            * (self.A2 - y)
            * self.sigmoid_derivative(self.A2)
        )

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradients
        dZ1 = (
            np.dot(dZ2, self.W2.T)
            * self.sigmoid_derivative(self.A1)
        )

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=5000):

        self.loss_history = []

        # Record initial loss at epoch 0
        predictions = self.forward(X)
        initial_loss = self.mse(y, predictions)

        self.loss_history.append(initial_loss)

        print(f"Epoch 0 - Loss: {initial_loss:.6f}")

        for epoch in range(1, epochs + 1):

            # Backward propagation
            self.backward(X, y)

            # Forward propagation with updated weights
            predictions = self.forward(X)

            # Calculate loss
            loss = self.mse(y, predictions)

            # Store loss
            self.loss_history.append(loss)

            # Print training progress
            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"- Loss: {loss:.6f}"
                )

        return self.loss_history

    def predict(self, X):

        predictions = self.forward(X)

        return (predictions >= 0.5).astype(int)