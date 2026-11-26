import unittest
import numpy as np

from neural_network import NeuralNetwork

from data_preprocessing import (
    clean_dataset,
    train_test_split,
    standardize_data
)

from evaluation import evaluate_model


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):

        self.model = NeuralNetwork(
            input_size=4,
            hidden_size=8,
            output_size=1,
            learning_rate=0.1,
            seed=42
        )

        self.X = np.array([
            [1.0, 0.5, -0.5, 1.0],
            [-1.0, -0.5, 0.5, -1.0],
            [0.5, 1.0, -1.0, 0.5],
            [-0.5, -1.0, 1.0, -0.5]
        ])

        self.y = np.array([
            [1],
            [0],
            [1],
            [0]
        ])

    def test_sigmoid(self):

        self.assertAlmostEqual(
            self.model.sigmoid(0),
            0.5
        )

        self.assertAlmostEqual(
            self.model.sigmoid(2),
            0.880797,
            places=6
        )

    def test_sigmoid_derivative(self):

        result = self.model.sigmoid_derivative(0.5)

        self.assertAlmostEqual(result, 0.25)

    def test_mse(self):

        y_true = np.array([[0], [1]])
        y_pred = np.array([[0.2], [0.8]])

        loss = self.model.mse(y_true, y_pred)

        self.assertAlmostEqual(loss, 0.04)

    def test_parameter_shapes(self):

        self.assertEqual(self.model.W1.shape, (4, 8))
        self.assertEqual(self.model.b1.shape, (1, 8))
        self.assertEqual(self.model.W2.shape, (8, 1))
        self.assertEqual(self.model.b2.shape, (1, 1))

    def test_forward_propagation(self):

        outputs = self.model.forward(self.X)

        self.assertEqual(outputs.shape, (4, 1))

        self.assertTrue(
            np.all((outputs >= 0) & (outputs <= 1))
        )

    def test_backpropagation(self):

        old_W1 = self.model.W1.copy()
        old_W2 = self.model.W2.copy()

        self.model.forward(self.X)

        self.model.backward(self.X, self.y)

        self.assertFalse(
            np.array_equal(old_W1, self.model.W1)
        )

        self.assertFalse(
            np.array_equal(old_W2, self.model.W2)
        )

    def test_training_loss_decreases(self):

        loss_history = self.model.train(
            self.X,
            self.y,
            epochs=100
        )

        self.assertLess(
            loss_history[-1],
            loss_history[0]
        )

        self.assertEqual(len(loss_history), 101)

    def test_prediction(self):

        predictions = self.model.predict(self.X)

        self.assertEqual(predictions.shape, (4, 1))

        self.assertTrue(
            np.all(np.isin(predictions, [0, 1]))
        )


class TestDataPreprocessing(unittest.TestCase):

    def test_duplicate_removal(self):

        data = np.array([
            [1, 2, 0],
            [1, 2, 0],
            [3, 4, 1]
        ])

        cleaned_data = clean_dataset(data)

        self.assertEqual(len(cleaned_data), 2)

    def test_train_test_split(self):

        X = np.arange(40).reshape(10, 4)
        y = np.arange(10) % 2

        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_indices,
            test_indices
        ) = train_test_split(X, y)

        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)

        overlap = np.intersect1d(
            train_indices,
            test_indices
        )

        self.assertEqual(len(overlap), 0)

    def test_standardization(self):

        X_train = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        X_test = np.array([
            [7.0, 8.0]
        ])

        train_scaled, test_scaled = standardize_data(
            X_train,
            X_test
        )

        np.testing.assert_allclose(
            np.mean(train_scaled, axis=0),
            [0, 0],
            atol=1e-10
        )

        np.testing.assert_allclose(
            np.std(train_scaled, axis=0),
            [1, 1],
            atol=1e-10
        )

        expected_test = (
            X_test - np.mean(X_train, axis=0)
        ) / np.std(X_train, axis=0)

        np.testing.assert_allclose(
            test_scaled,
            expected_test
        )


class TestEvaluation(unittest.TestCase):

    def test_evaluation_metrics(self):

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = evaluate_model(y_true, y_pred)

        self.assertAlmostEqual(
            metrics["accuracy"],
            0.75
        )

        self.assertAlmostEqual(
            metrics["precision"],
            2 / 3
        )

        self.assertAlmostEqual(
            metrics["recall"],
            1.0
        )

        self.assertAlmostEqual(
            metrics["f1_score"],
            0.8
        )

        np.testing.assert_array_equal(
            metrics["confusion_matrix"],
            np.array([
                [1, 1],
                [0, 2]
            ])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)