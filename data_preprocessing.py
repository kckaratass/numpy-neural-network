import numpy as np


def load_dataset(file_path):
    """Load the original dataset."""

    data = np.loadtxt(file_path, delimiter=",")

    return data


def clean_dataset(raw_data):
    """Remove duplicate records while preserving row order."""

    _, unique_indices = np.unique(
        raw_data,
        axis=0,
        return_index=True
    )

    data = raw_data[np.sort(unique_indices)]

    print("\nData cleaning results:")
    print("Original records:", len(raw_data))
    print("Removed duplicates:", len(raw_data) - len(data))
    print("Remaining records:", len(data))

    return data


def inspect_dataset(data):
    """Display dataset information."""

    print("\nDataset information:")
    print("Dataset shape:", data.shape)
    print("Missing values:", np.isnan(data).sum())

    X = data[:, :-1]
    y = data[:, -1]

    classes, counts = np.unique(y, return_counts=True)

    print("Input shape:", X.shape)
    print("Target shape:", y.shape)
    print("Classes:", classes)
    print("Class distribution:", counts)


def split_features_target(data):
    """Separate input features and target variable."""

    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def train_test_split(X, y, train_ratio=0.8, seed=42):
    """Randomly split data into training and test sets."""

    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))

    train_size = int(len(X) * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    print("\nTraining and test split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\nTraining class distribution:")
    print(np.unique(y_train, return_counts=True))

    print("\nTest class distribution:")
    print(np.unique(y_test, return_counts=True))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_indices,
        test_indices
    )


def standardize_data(X_train, X_test):
    """Standardize data using training statistics only."""

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Prevent division by zero
    std = np.where(std == 0, 1, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("\nStandardization results:")

    print("Training mean:")
    print(np.mean(X_train_scaled, axis=0))

    print("Training standard deviation:")
    print(np.std(X_train_scaled, axis=0))

    return X_train_scaled, X_test_scaled


def check_data_quality(data, train_indices, test_indices):
    """Check duplicates and train-test overlap."""

    unique_rows = np.unique(data, axis=0)

    duplicate_count = len(data) - len(unique_rows)

    overlap = np.intersect1d(
        train_indices,
        test_indices
    )

    train_rows = set(map(tuple, data[train_indices]))
    test_rows = set(map(tuple, data[test_indices]))

    shared_rows = train_rows.intersection(test_rows)

    train_features = set(
        map(tuple, data[train_indices, :-1])
    )

    test_features = set(
        map(tuple, data[test_indices, :-1])
    )

    shared_features = train_features.intersection(
        test_features
    )

    print("\nData quality analysis:")
    print("Missing values:", np.isnan(data).sum())
    print("Remaining duplicates:", duplicate_count)
    print("Overlapping indices:", len(overlap))
    print("Identical records:", len(shared_rows))
    print("Identical input features:", len(shared_features))


def prepare_data(file_path, train_ratio=0.8, seed=42):
    """Execute the complete data preprocessing pipeline."""

    raw_data = load_dataset(file_path)

    data = clean_dataset(raw_data)

    inspect_dataset(data)

    X, y = split_features_target(data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_indices,
        test_indices
    ) = train_test_split(
        X,
        y,
        train_ratio=train_ratio,
        seed=seed
    )

    check_data_quality(
        data,
        train_indices,
        test_indices
    )

    X_train, X_test = standardize_data(
        X_train,
        X_test
    )

    # Reshape targets for neural network
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test