import numpy as np
from pathlib import Path


def save_results(
    loss_history,
    y_test,
    test_outputs,
    test_predictions,
    metrics,
    initial_accuracy,
    train_accuracy,
    initial_loss,
    train_loss,
    test_loss
):

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save training loss
    epochs = np.arange(len(loss_history))

    training_loss_data = np.column_stack(
        (epochs, loss_history)
    )

    np.savetxt(
        results_dir / "training_loss.csv",
        training_loss_data,
        delimiter=",",
        header="epoch,mse",
        comments="",
        fmt=["%d", "%.10f"]
    )

    # Save test predictions
    sample_ids = np.arange(1, len(y_test) + 1)

    correct_predictions = (
        y_test.ravel() == test_predictions.ravel()
    ).astype(int)

    prediction_data = np.column_stack((
        sample_ids,
        y_test.ravel(),
        test_outputs.ravel(),
        test_predictions.ravel(),
        correct_predictions
    ))

    np.savetxt(
        results_dir / "test_predictions.csv",
        prediction_data,
        delimiter=",",
        header="sample,actual,output,predicted,correct",
        comments="",
        fmt=["%d", "%d", "%.10f", "%d", "%d"]
    )

    # Save performance metrics
    metric_data = np.array([
        ["Initial Accuracy", initial_accuracy],
        ["Training Accuracy", train_accuracy],
        ["Test Accuracy", metrics["accuracy"]],
        ["Precision", metrics["precision"]],
        ["Recall", metrics["recall"]],
        ["F1-Score", metrics["f1_score"]],
        ["Initial MSE", initial_loss],
        ["Training MSE", train_loss],
        ["Test MSE", test_loss]
    ], dtype=object)

    np.savetxt(
        results_dir / "metrics.csv",
        metric_data,
        delimiter=",",
        header="metric,value",
        comments="",
        fmt=["%s", "%.10f"]
    )

    # Save confusion matrix
    np.savetxt(
        results_dir / "confusion_matrix.csv",
        metrics["confusion_matrix"],
        delimiter=",",
        header="Predicted_0,Predicted_1",
        comments="",
        fmt="%d"
    )

    print("\nResults saved successfully:")
    print("results/training_loss.csv")
    print("results/test_predictions.csv")
    print("results/metrics.csv")
    print("results/confusion_matrix.csv")