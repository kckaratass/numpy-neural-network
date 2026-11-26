import numpy as np

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