import numpy as np

def generate_data(NA, alpha):
    class_A = np.random.multivariate_normal([0, 0], np.eye(2), NA)
    NB = int(alpha * NA)
    class_B = np.random.multivariate_normal([2, 2], np.eye(2), NB)
    X = np.vstack([class_A, class_B])
    y = np.hstack([np.zeros(NA), np.ones(NB)])
    return X, y

NA = 100
alphas = [1, 10, 100, 1000]

for alpha in alphas:
    X, y = generate_data(NA, alpha)
    np.save(f"data_X_{alpha}.npy", X)
    np.save(f"data_y_{alpha}.npy", y)
    print(f"Data generated and saved for α = {alpha}")
