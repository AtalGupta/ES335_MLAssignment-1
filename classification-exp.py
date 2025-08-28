import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X_df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
y_df = pd.Series(y)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Generated Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 2a: Show the usage of your decision tree on the above dataset. The first 70% of the data should be used for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall of the decision tree you implemented on the test dataset. 
# data split
split_index = int(0.7 * len(X_df))
X_train, X_test = X_df[:split_index], X_df[split_index:]
y_train, y_test = y_df[:split_index], y_df[split_index:]

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)
print("Classes:", np.unique(y_train))

# Train the decision tree
tree = DecisionTree(criterion="information_gain", max_depth=3)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)

# Evaluate the model
acc_score = accuracy(y_pred, y_test)
unique_class = sorted(y_df.unique())
precision_metrics = {}
recall_metrics = {}

print(f"Accuracy: {acc_score:.4f}")

for cls in unique_class:
    precision_metrics[cls] = precision(y_pred, y_test, cls=cls)
    recall_metrics[cls] = recall(y_pred, y_test, cls=cls)
    print(f"Class {cls} - Precision: {precision_metrics[cls]:.4f}, Recall: {recall_metrics[cls]:.4f}")
    
    
#2b: Use 5 fold cross-validation from scratch in python on the dataset. Using nested cross-validation find the optimum depth of the tree

def create_folds(X, y, n_folds=5):
    # Reset indices to ensure they start from 0 and are continuous
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
        
    indices_per_class = {}
    for cls in np.unique(y):
        indices_per_class[cls] = y[y == cls].index.tolist()

    folds = [[] for _ in range(n_folds)]
    for cls, indices in indices_per_class.items():
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds
        for i in range(n_folds):
            start = i * fold_size
            if i == n_folds - 1:
                end = len(indices)
            else:
                end = (i + 1) * fold_size
            folds[i].extend(indices[start:end])

    for fold in folds:
        np.random.shuffle(fold)
    return folds

def cross_validate_depth(X, y, depths, criterion="information_gain", n_folds=5):
    # Reset indices to ensure they start from 0
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    folds = create_folds(X, y, n_folds=n_folds)
    cv_results = {depth: [] for depth in depths}
    
    for fold_idx in range(n_folds):
        val_indices = folds[fold_idx]
        train_indices = []
        for i in range(n_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])

        X_fold_train = X.iloc[train_indices]
        y_fold_train = y.iloc[train_indices]
        X_fold_val = X.iloc[val_indices]
        y_fold_val = y.iloc[val_indices]

        for depth in depths:
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(X_fold_train, y_fold_train)
            y_pred = tree.predict(X_fold_val)
            fold_acc = accuracy(y_pred, y_fold_val)
            cv_results[depth].append(fold_acc)

    return cv_results


def nested_cross_validation(X, y, depths, criterion='information_gain', n_folds=5):
    # Reset indices to ensure they start from 0
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    outer_fold_results = []
    optimal_depths = []
    outer_folds_indices = create_folds(X, y, n_folds=n_folds)

    for outer_fold in range(n_folds):
        test_indices = outer_folds_indices[outer_fold]
        train_val_indices = []
        for i in range(n_folds):
            if i != outer_fold:
                train_val_indices.extend(outer_folds_indices[i])
        X_train_val = X.iloc[train_val_indices]
        y_train_val = y.iloc[train_val_indices]
        X_test_outer = X.iloc[test_indices]
        y_test_outer = y.iloc[test_indices]
        inner_cv_results = cross_validate_depth(X_train_val, y_train_val, depths, criterion=criterion, n_folds=3) 
        
        depth_means = {}
        for depth in depths:
            depth_means[depth] = np.mean(inner_cv_results[depth])
            
        optimal_depth = max(depth_means.keys(), key=(lambda k: depth_means[k]))
        optimal_depths.append(optimal_depth)
        
        final_model = DecisionTree(criterion=criterion, max_depth=optimal_depth)
        final_model.fit(X_train_val, y_train_val)
        
        y_pred_outer = final_model.predict(X_test_outer)
        outer_accuracy = accuracy(y_pred_outer, y_test_outer)
        outer_fold_results.append(outer_accuracy)

    return outer_fold_results, optimal_depths

depths = [1, 2, 3, 4, 5, 6, 7, 8]
outer_fold_results, optimal_depths = nested_cross_validation(X_df, y_df, depths)

print(f"\n2b) Nested Cross-Validation Results:")
print(f"Optimal depths across outer folds: {optimal_depths}")
print(f"Most common optimal depth: {max(set(optimal_depths), key=optimal_depths.count)}")
print(f"Outer fold accuracies: {[f'{acc:.4f}' for acc in outer_fold_results]}")
print(f"Mean outer CV accuracy: {np.mean(outer_fold_results):.4f} ± {np.std(outer_fold_results):.4f}")

# Simple CV for comparison
simple_cv_results = cross_validate_depth(X_df, y_df, depths, "information_gain", 5)
depth_stats = {}
for depth in depths:
    scores = simple_cv_results[depth]
    depth_stats[depth] = {'mean': np.mean(scores), 'std': np.std(scores)}

optimal_depth_simple = max(depth_stats.keys(), key=lambda d: depth_stats[d]['mean'])
print(f"\nSimple 5-fold CV optimal depth: {optimal_depth_simple}")
print(f"Simple CV accuracy: {depth_stats[optimal_depth_simple]['mean']:.4f} ± {depth_stats[optimal_depth_simple]['std']:.4f}")

print("\nExperiment completed!")

