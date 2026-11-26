import numpy as np
from neural_network import NeuralNetwork

# Load dataset
data = np.loadtxt(
    "data/data_banknote_authentication.txt",
    delimiter=","
)

# Inspect dataset
print("Dataset shape:", data.shape)
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