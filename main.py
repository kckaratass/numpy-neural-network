import numpy as np

from data_preprocessing import prepare_data
from neural_network import NeuralNetwork
from evaluation import evaluate_model
from save_results import save_results


# Experiment configuration
DATA_PATH = "data/data_banknote_authentication.txt"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 5000


def main():

    print("=" * 60)
    print("BANKNOTE AUTHENTICATION - NUMPY NEURAL NETWORK")
    print("=" * 60)

    # 1. Prepare dataset
    X_train, X_test, y_train, y_test = prepare_data(
        file_path=DATA_PATH,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED
    )

    # 2. Initialize neural network
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED
    )

    total_parameters = (
        model.W1.size
        + model.b1.size
        + model.W2.size
        + model.b2.size
    )

    print("\nNeural network configuration:")
    print("Input neurons:", X_train.shape[1])
    print("Hidden neurons:", HIDDEN_SIZE)
    print("Output neurons:", OUTPUT_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", EPOCHS)
    print("Total parameters:", total_parameters)

    # 3. Evaluate initial model
    initial_outputs = model.forward(X_train)

    initial_loss = model.mse(
        y_train,
        initial_outputs
    )

    initial_predictions = model.predict(X_train)

    initial_accuracy = np.mean(
        initial_predictions == y_train
    )

    print("\nInitial model performance:")
    print(f"Initial MSE: {initial_loss:.6f}")
    print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")

    # 4. Train neural network
    print("\nModel training started:\n")

    loss_history = model.train(
        X_train,
        y_train,
        epochs=EPOCHS
    )

    print("\nModel training completed.")

    # 5. Evaluate training performance
    train_outputs = model.forward(X_train)

    train_loss = model.mse(
        y_train,
        train_outputs
    )

    train_predictions = model.predict(X_train)

    train_accuracy = np.mean(
        train_predictions == y_train
    )

    loss_reduction = (
        (initial_loss - train_loss)
        / initial_loss
    ) * 100

    print("\nTraining results:")
    print(f"Initial MSE: {initial_loss:.6f}")
    print(f"Final MSE: {train_loss:.6f}")
    print(f"MSE reduction: {loss_reduction:.2f}%")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")

    # 6. Evaluate test performance
    test_outputs = model.forward(X_test)

    test_predictions = model.predict(X_test)

    test_loss = model.mse(
        y_test,
        test_outputs
    )

    metrics = evaluate_model(
        y_test,
        test_predictions
    )

    print("\nTest results:")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1-score: {metrics['f1_score'] * 100:.2f}%")

    # 7. Display confusion matrix
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()

    print("\nPrediction summary:")
    print("Total test samples:", len(y_test))
    print("Correct predictions:", tn + tp)
    print("Incorrect predictions:", fp + fn)

    # 8. Compare training and test accuracy
    accuracy_gap = (
        train_accuracy - metrics["accuracy"]
    ) * 100

    print("\nTraining and test comparison:")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Accuracy gap: {accuracy_gap:.4f} percentage points")

    # 9. Display sample predictions
    print("\nFirst 10 test predictions:")
    print("Actual | Output   | Predicted")

    for i in range(min(10, len(y_test))):

        actual = int(y_test[i, 0])
        output = test_outputs[i, 0]
        predicted = int(test_predictions[i, 0])

        print(
            f"{actual:6d} | "
            f"{output:.6f} | "
            f"{predicted:9d}"
        )

    # 10. Save experiment results
    save_results(
        loss_history=loss_history,
        y_test=y_test,
        test_outputs=test_outputs,
        test_predictions=test_predictions,
        metrics=metrics,
        initial_accuracy=initial_accuracy,
        train_accuracy=train_accuracy,
        initial_loss=initial_loss,
        train_loss=train_loss,
        test_loss=test_loss
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()