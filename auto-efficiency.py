import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Remove car name column as it's not useful for prediction
data = data.drop('car name', axis=1)

# Handle missing values in horsepower (replace '?' with NaN and then drop)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data = data.dropna()

# Prepare features and target
X = data.drop('mpg', axis=1)
y = data['mpg']

# Split data into train and test sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# a) Show the usage of your decision tree for the automotive efficiency problem
print("a) Custom Decision Tree Implementation")
print("="*50)

# Train our custom decision tree
custom_tree = DecisionTree(criterion="information_gain", max_depth=5)
custom_tree.fit(X_train, y_train)

# Make predictions
y_pred_custom = custom_tree.predict(X_test)

# Calculate metrics
rmse_custom = rmse(y_pred_custom, y_test)
mae_custom = mae(y_pred_custom, y_test)

print(f"Custom Decision Tree Results:")
print(f"RMSE: {rmse_custom:.4f}")
print(f"MAE: {mae_custom:.4f}")

# b) Compare the performance of your model with the decision tree module from scikit learn
print("\nb) Comparison with Scikit-learn Decision Tree")
print("="*50)

# Train scikit-learn decision tree with similar parameters
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)

# Make predictions
y_pred_sklearn = sklearn_tree.predict(X_test)

# Calculate metrics
rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)

print(f"Scikit-learn Decision Tree Results:")
print(f"RMSE: {rmse_sklearn:.4f}")
print(f"MAE: {mae_sklearn:.4f}")

# Performance comparison
print(f"\nPerformance Comparison:")
print(f"{'Metric':<10} {'Custom Tree':<15} {'Scikit-learn':<15}")
print("-" * 45)
print(f"{'RMSE':<10} {rmse_custom:<15.4f} {rmse_sklearn:<15.4f}")
print(f"{'MAE':<10} {mae_custom:<15.4f} {mae_sklearn:<15.4f}")

# Visualization of predictions vs actual values
plt.figure(figsize=(12, 4))

# Custom tree predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_custom, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Custom Tree (RMSE: {rmse_custom:.3f})')
plt.grid(True, alpha=0.3)

# Scikit-learn predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_sklearn, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Scikit-learn Tree (RMSE: {rmse_sklearn:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()