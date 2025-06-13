import numpy as np

def load_data(alpha):
    X = np.load(f"data_X_{alpha}.npy")
    y = np.load(f"data_y_{alpha}.npy")
    return X, y

def ml_classifier(X_train, y_train, X_test):
    mean_A = np.mean(X_train[y_train == 0], axis=0)
    mean_B = np.mean(X_train[y_train == 1], axis=0)
    cov_A = np.cov(X_train[y_train == 0], rowvar=False)
    cov_B = np.cov(X_train[y_train == 1], rowvar=False)
    inv_cov_A = np.linalg.inv(cov_A)
    inv_cov_B = np.linalg.inv(cov_B)
    dist_A = np.array([np.sqrt((x - mean_A).T @ inv_cov_A @ (x - mean_A)) for x in X_test])
    dist_B = np.array([np.sqrt((x - mean_B).T @ inv_cov_B @ (x - mean_B)) for x in X_test])
    predictions = np.where(dist_A < dist_B, 0, 1)
    return predictions

def map_classifier(X_train, y_train, X_test):
    prior_A = np.mean(y_train == 0)
    prior_B = 1 - prior_A
    mean_A = np.mean(X_train[y_train == 0], axis=0)
    mean_B = np.mean(X_train[y_train == 1], axis=0)
    cov_A = np.cov(X_train[y_train == 0], rowvar=False)
    cov_B = np.cov(X_train[y_train == 1], rowvar=False)
    inv_cov_A = np.linalg.inv(cov_A)
    inv_cov_B = np.linalg.inv(cov_B)
    dist_A = np.array([np.sqrt((x - mean_A).T @ inv_cov_A @ (x - mean_A)) for x in X_test])
    dist_B = np.array([np.sqrt((x - mean_B).T @ inv_cov_B @ (x - mean_B)) for x in X_test])
    posterior_A = dist_A - 2 * np.log(prior_A)
    posterior_B = dist_B - 2 * np.log(prior_B)
    predictions = np.where(posterior_A < posterior_B, 0, 1)
    return predictions

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_classifiers():
    alpha_values = [1, 10, 100, 1000]
    for alpha in alpha_values:
        print(f"Evaluating for α = {alpha}...")
        X, y = load_data(alpha)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        ml_predictions = ml_classifier(X_train, y_train, X_test)
        ml_accuracy = accuracy(y_test, ml_predictions)
        map_predictions = map_classifier(X_train, y_train, X_test)
        map_accuracy = accuracy(y_test, map_predictions)
        print(f"ML Accuracy: {ml_accuracy:.4f}")
        print(f"MAP Accuracy: {map_accuracy:.4f}")
        print()

if __name__ == "__main__":
    evaluate_classifiers()
