import numpy as np

def load_data(alpha):
    X = np.load(f"data_X_{alpha}.npy")
    y = np.load(f"data_y_{alpha}.npy")
    return X, y

alpha = 1000
X, y = load_data(alpha)

print("Loaded X data:")
print(X)
print("\nLoaded y data:")
print(y)
