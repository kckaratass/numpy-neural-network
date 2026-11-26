import numpy as np
from neural_network import NeuralNetwork
from evaluation import evaluate_model


# Load original dataset
raw_data = np.loadtxt(
    "data/data_banknote_authentication.txt",
    delimiter=","
)

# Remove duplicate records while preserving row order
_, unique_indices = np.unique(
    raw_data,
    axis=0,
    return_index=True
)

data = raw_data[np.sort(unique_indices)]

# Display data cleaning results
print("Data cleaning results:")
print("Original records:", len(raw_data))
print("Removed duplicates:", len(raw_data) - len(data))
print("Remaining records:", len(data))


# Inspect dataset
print("\nDataset shape:", data.shape)
print("\nFirst 5 samples:")
print(data[:5])


# Separate features and target
X = data[:, :-1]
y = data[:, -1]

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)


# Check missing values
print("\nMissing values:", np.isnan(data).sum())


# Check class distribution
classes, counts = np.unique(y, return_counts=True)

print("Classes:", classes)
print("Class distribution:", counts)


# Random train-test split
rng = np.random.default_rng(42)

indices = rng.permutation(len(X))

train_size = int(len(X) * 0.8)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("\nTraining and test split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Check class distribution after splitting
print("\nTraining class distribution:")
print(np.unique(y_train, return_counts=True))

print("\nTest class distribution:")
print(np.unique(y_test, return_counts=True))


# Check for overlapping indices
overlap = np.intersect1d(train_indices, test_indices)

print("\nOverlapping samples:", len(overlap))


# Calculate statistics using training data only
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)


# Prevent division by zero
std = np.where(std == 0, 1, std)


# Standardize training and test data
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Inspect standardization results
print("\nStandardization results:")

print("Original training mean:", mean)
print("Original training std:", std)

print("\nStandardized training mean:")
print(np.mean(X_train, axis=0))

print("\nStandardized training std:")
print(np.std(X_train, axis=0))

print("\nStandardized test mean:")
print(np.mean(X_test, axis=0))

print("\nStandardized test std:")
print(np.std(X_test, axis=0))

print("\nFirst 5 standardized training samples:")
print(X_train[:5])


# Initialize neural network
model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=8,
    output_size=1,
    learning_rate=0.1,
    seed=42
)

print("\nNeural network architecture:")
print("W1 shape:", model.W1.shape)
print("b1 shape:", model.b1.shape)
print("W2 shape:", model.W2.shape)
print("b2 shape:", model.b2.shape)

total_parameters = (
    model.W1.size
    + model.b1.size
    + model.W2.size
    + model.b2.size
)

print("Total parameters:", total_parameters)


# Test sigmoid function
print("\nSigmoid test:")
print("Sigmoid(-2):", model.sigmoid(-2))
print("Sigmoid(0):", model.sigmoid(0))
print("Sigmoid(2):", model.sigmoid(2))


# Reshape target arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# Forward propagation
predictions = model.forward(X_train)

print("\nForward propagation results:")
print("Z1 shape:", model.Z1.shape)
print("A1 shape:", model.A1.shape)
print("Z2 shape:", model.Z2.shape)
print("A2 shape:", model.A2.shape)

print("\nFirst 5 predictions:")
print(predictions[:5])

print("\nFirst 5 actual values:")
print(y_train[:5])


# Calculate initial loss
initial_loss = model.mse(y_train, predictions)

print("\nInitial MSE:", initial_loss)


# Calculate initial accuracy
predicted_classes = (predictions >= 0.5).astype(int)

initial_accuracy = np.mean(
    predicted_classes == y_train
)

print("Initial accuracy:", initial_accuracy * 100, "%")


# Test backpropagation
print("\nBackpropagation test:")


# Save initial weights
old_W1 = model.W1.copy()
old_W2 = model.W2.copy()


# Forward propagation before update
predictions_before = model.forward(X_train)

loss_before = model.mse(
    y_train, predictions_before
)


# Perform one backward propagation
model.backward(X_train, y_train)


# Forward propagation after update
predictions_after = model.forward(X_train)

loss_after = model.mse(
    y_train, predictions_after
)

print("Loss before update:", loss_before)
print("Loss after update:", loss_after)
print("Loss reduction:", loss_before - loss_after)

print("\nWeight update check:")

print(
    "W1 changed:",
    not np.array_equal(old_W1, model.W1)
)

print(
    "W2 changed:",
    not np.array_equal(old_W2, model.W2)
)

print(
    "Loss decreased:",
    loss_after < loss_before
)


# Reinitialize model before training
model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=8,
    output_size=1,
    learning_rate=0.1,
    seed=42
)


# Train neural network
print("\nModel training started:\n")

loss_history = model.train(
    X_train,
    y_train,
    epochs=5000
)

print("\nModel training completed.")


# Evaluate training performance
train_outputs = model.forward(X_train)
train_loss = model.mse(y_train, train_outputs)

train_predictions = model.predict(X_train)

train_accuracy = np.mean(
    train_predictions == y_train
)


# Calculate loss improvement
loss_reduction = (
    (initial_loss - train_loss) / initial_loss
) * 100

print("\nTraining results:")
print(f"Initial MSE: {initial_loss:.6f}")
print(f"Final MSE: {train_loss:.6f}")
print(f"MSE reduction: {loss_reduction:.2f}%")

print(f"Initial accuracy: {initial_accuracy * 100:.2f}%")
print(f"Final training accuracy: {train_accuracy * 100:.2f}%")

print("Recorded loss values:", len(loss_history))


# Evaluate model on test data
print("\nTest evaluation:")

test_outputs = model.forward(X_test)

test_predictions = model.predict(X_test)

test_loss = model.mse(y_test, test_outputs)


# Calculate performance metrics
metrics = evaluate_model(
    y_test,
    test_predictions
)

print(f"Test MSE: {test_loss:.6f}")
print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Precision: {metrics['precision'] * 100:.2f}%")
print(f"Recall: {metrics['recall'] * 100:.2f}%")
print(f"F1-score: {metrics['f1_score'] * 100:.2f}%")


# Display confusion matrix
print("\nConfusion matrix:")
print(metrics["confusion_matrix"])

tn, fp, fn, tp = metrics["confusion_matrix"].ravel()

print("\nConfusion matrix details:")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)


# Correct and incorrect predictions
print("\nPrediction summary:")
print("Total test samples:", len(y_test))
print("Correct predictions:", tp + tn)
print("Incorrect predictions:", fp + fn)


# Compare training and test accuracy
accuracy_gap = (
    train_accuracy - metrics["accuracy"]
) * 100

print("\nTraining and test comparison:")
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Accuracy gap: {accuracy_gap:.2f} percentage points")


# Compare actual and predicted values
print("\nFirst 10 test predictions:")

print("Actual | Output   | Predicted")

for i in range(10):

    actual = int(y_test[i, 0])
    output = test_outputs[i, 0]
    predicted = int(test_predictions[i, 0])

    print(
        f"{actual:6d} | "
        f"{output:.6f} | "
        f"{predicted:9d}"
    )


# Final data quality checks
unique_rows = np.unique(data, axis=0)

duplicate_count = len(data) - len(unique_rows)

print("\nFinal data quality analysis:")
print("Total records:", len(data))
print("Unique records:", len(unique_rows))
print("Remaining duplicates:", duplicate_count)


# Check identical records across train and test sets
train_rows = set(map(tuple, data[train_indices]))
test_rows = set(map(tuple, data[test_indices]))

shared_rows = train_rows.intersection(test_rows)

print(
    "Identical records across train and test:",
    len(shared_rows)
)


# Check identical input features across train and test sets
train_features = set(
    map(tuple, data[train_indices, :-1])
)

test_features = set(
    map(tuple, data[test_indices, :-1])
)

shared_features = train_features.intersection(
    test_features
)

print(
    "Identical input features across train and test:",
    len(shared_features)
)