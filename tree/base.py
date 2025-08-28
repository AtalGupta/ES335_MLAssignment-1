"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, left=None, right=None, depth=0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.depth = depth

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"] # criterion won't be used for regression
    max_depth: int = 5  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Determine if this is a regression or classification problem
        self.is_regression = check_ifreal(y)
        
        # For regression, we use MSE; for classification, use the specified criterion
        if self.is_regression:
            criterion_str = 'mse'
        else:
            criterion_str = 'entropy' if self.criterion == 'information_gain' else 'gini'
        
        # Build the tree recursively
        self.root = self._build_tree(X, y, criterion_str, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, criterion: str, depth: int) -> Node:
        """
        Recursively build the decision tree
        """
        # Base cases for stopping
        # 1. Maximum depth reached
        if depth >= self.max_depth:
            return Node(is_leaf=True, prediction=leaf_value(y), depth=depth)
        
        # 2. All labels are the same (pure node)
        if is_pure(y):
            return Node(is_leaf=True, prediction=leaf_value(y), depth=depth)
        
        # 3. No features left or insufficient data
        if X.empty or len(y) <= 1:
            return Node(is_leaf=True, prediction=leaf_value(y), depth=depth)
        
        # Find the best feature and split value
        features = pd.Series(X.columns)
        best_feature, best_split_value = opt_split_attribute(
            X, y, criterion, features, min_samples_leaf=1
        )
        
        # If no good split found, create leaf
        if best_feature is None:
            return Node(is_leaf=True, prediction=leaf_value(y), depth=depth)
        
        # Split the data
        (X_left, y_left), (X_right, y_right) = split_data(X, y, best_feature, best_split_value)
        
        # If split results in empty sets, create leaf
        if X_left.empty or X_right.empty:
            return Node(is_leaf=True, prediction=leaf_value(y), depth=depth)
        
        # Create internal node
        node = Node(
            is_leaf=False,
            feature=best_feature,
            threshold=best_split_value,
            depth=depth
        )
        
        # Recursively build left and right subtrees
        node.left = self._build_tree(X_left, y_left, criterion, depth + 1)
        node.right = self._build_tree(X_right, y_right, criterion, depth + 1)
        
        return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        predictions = []
        for idx, row in X.iterrows():
            prediction = self._predict_single(row, self.root)
            predictions.append(prediction)
        
        return pd.Series(predictions, index=X.index)

    def _predict_single(self, sample: pd.Series, node: Node):
        """
        Predict a single sample by traversing the tree
        """
        # If leaf node, return prediction
        if node.is_leaf:
            return node.prediction
        
        # Get the feature value for this sample
        feature_value = sample[node.feature]
        
        # Decide which branch to follow
        if pd.api.types.is_numeric_dtype(pd.Series([feature_value])):
            # Numeric feature: use threshold
            if pd.isna(feature_value):
                # Handle missing values - go to left by default
                return self._predict_single(sample, node.left)
            elif feature_value < node.threshold:
                return self._predict_single(sample, node.left)
            else:
                return self._predict_single(sample, node.right)
        else:
            # Categorical feature: exact match
            if feature_value == node.threshold:
                return self._predict_single(sample, node.left)
            else:
                return self._predict_single(sample, node.right)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.root is None:
            print("Tree has not been fitted yet.")
            return
        
        print("Decision Tree Structure:")
        print("=" * 50)
        self._print_tree(self.root, prefix="", is_left=True, is_root=True)

    def _print_tree(self, node: Node, prefix: str = "", is_left: bool = True, is_root: bool = False):
        """
        Recursively print the tree structure
        """
        if node is None:
            return
        
        if node.is_leaf:
            # Leaf node - show prediction
            if self.is_regression:
                print(f"{prefix}Predict: {node.prediction:.3f}")
            else:
                print(f"{prefix}Predict: {node.prediction}")
        else:
            # Internal node - show split condition
            if pd.api.types.is_numeric_dtype(pd.Series([node.threshold])):
                condition = f"{node.feature} < {node.threshold:.3f}"
            else:
                condition = f"{node.feature} == {node.threshold}"
            
            if is_root:
                print(f"?({condition})")
            else:
                print(f"{prefix}?({condition})")
            
            # Print left subtree (True/Yes branch)
            left_prefix = prefix + ("    " if is_root else "│   ")
            print(f"{prefix}├── Yes:")
            self._print_tree(node.left, left_prefix, True, False)
            
            # Print right subtree (False/No branch)
            print(f"{prefix}└── No:")
            right_prefix = prefix + ("    " if is_root else "    ")
            self._print_tree(node.right, right_prefix, False, False)



