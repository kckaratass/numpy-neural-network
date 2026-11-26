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

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)