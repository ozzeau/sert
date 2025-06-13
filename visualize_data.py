import numpy as np
import matplotlib.pyplot as plt
from classifiers import ml_classifier, map_classifier
from load_data import load_data
from classifiers import accuracy

def plot_decision_boundary(X, y, classifier, alpha, title="Decision Boundary"):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = classifier(X, y, grid_points)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(f'{title} (α={alpha})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

def plot_accuracy_vs_alpha():
    alpha_values = [1, 10, 100, 1000]
    ml_accuracies = []
    map_accuracies = []

    for alpha in alpha_values:
        X, y = load_data(alpha)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        ml_predictions = ml_classifier(X_train, y_train, X_test)
        ml_accuracy = accuracy(y_test, ml_predictions)
        ml_accuracies.append(ml_accuracy)

        map_predictions = map_classifier(X_train, y_train, X_test)
        map_accuracy = accuracy(y_test, map_predictions)
        map_accuracies.append(map_accuracy)

    plt.plot(alpha_values, ml_accuracies, label='ML Classifier', marker='o')
    plt.plot(alpha_values, map_accuracies, label='MAP Classifier', marker='o')
    plt.xscale('log')  # Use log scale for α
    plt.xlabel('Alpha (Class Imbalance Factor)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Alpha')
    plt.legend()
    plt.show()

def visualize():
    alpha_values = [1, 10, 100, 1000]
    for alpha in alpha_values:
        print(f"Visualizing decision boundaries for α={alpha}")
        X, y = load_data(alpha)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        plot_decision_boundary(X_train, y_train, ml_classifier, alpha, title="ML Classifier")
        plot_decision_boundary(X_train, y_train, map_classifier, alpha, title="MAP Classifier")

    plot_accuracy_vs_alpha()

if __name__ == "__main__":
    visualize()
