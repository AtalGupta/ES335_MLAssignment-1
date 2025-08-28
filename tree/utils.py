"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, drop_first=False)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    For regression: continuous numeric values
    For classification: discrete values (even if numeric like 0,1,2)
    """
    if not pd.api.types.is_numeric_dtype(y):
        return False  # Non-numeric is always discrete (classification)
    
    # Check if values appear to be discrete categories vs continuous
    unique_values = y.nunique()
    total_values = len(y)
    
    # If very few unique values relative to total, likely discrete categories
    if unique_values <= 10 or unique_values / total_values < 0.05:
        return False  # Discrete (classification)
    
    # Check if all values are integers (could still be discrete categories)
    if y.dtype in ['int64', 'int32'] and unique_values <= 20:
        return False  # Discrete (classification)
    
    return True  # Continuous (regression)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    assert isinstance(Y, pd.Series), "Input must be a pandas Series"
    
    if Y.empty:
        return 0.0
    
    value_counts = Y.value_counts()
    probs = value_counts / len(Y)
    entropy_val = 0.0
    
    for p in probs:
        if p > 0:
            entropy_val -= p * np.log2(p)

    return entropy_val

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    assert isinstance(Y, pd.Series), "Input must be a pandas Series"
    
    if Y.empty:
        return 0.0
    
    value_counts = Y.value_counts()
    probs = value_counts / len(Y)
    gini = 1 - np.sum(probs ** 2)
    
    return gini

def mse(Y: pd.Series) -> float:
    """
    Function to calculate mean squared error for regression
    """
    assert isinstance(Y, pd.Series), "Input must be a pandas Series"
    assert pd.api.types.is_numeric_dtype(Y), "MSE requires numeric data"
    
    if Y.empty:
        return 0.0
    
    return ((Y - Y.mean()) ** 2).mean()

def compute_impurity(Y: pd.Series, criterion: str) -> float:
    """
    Unified impurity calculator for entropy/gini/mse.
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    assert criterion in ['entropy', 'gini', 'mse'], f"Unknown criterion: {criterion}"
    
    if criterion == 'entropy':
        return entropy(Y)
    elif criterion == 'gini':
        return gini_index(Y)
    elif criterion == 'mse':
        return mse(Y)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    assert isinstance(attr, pd.Series), "attr must be a pandas Series"
    assert len(Y) == len(attr), "Y and attr must have the same length"
    
    if Y.empty or attr.empty:
        return 0.0
    
    parent_impurity = compute_impurity(Y, criterion)
    
    weighted_impurity = 0.0
    total_samples = len(Y)
    
    for val in attr.unique():
        if pd.isna(val):
            subset_mask = attr.isna()
        else:
            subset_mask = attr == val
            
        subset_Y = Y[subset_mask]
        
        if not subset_Y.empty:
            weight = len(subset_Y) / total_samples
            subset_impurity = compute_impurity(subset_Y, criterion)
            weighted_impurity += weight * subset_impurity
    
    gain = parent_impurity - weighted_impurity
    return max(0.0, gain)  # Ensure non-negative gain

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, 
                       features: pd.Series, min_samples_leaf: int = 1) -> Tuple[Optional[str], Optional[Union[float, str]]]:
    """
    Find the best attribute and split value to maximize information gain.
    
    Args:
        X: Input features DataFrame
        y: Target variable Series
        criterion: Splitting criterion ('entropy', 'gini', 'mse')
        features: Available features to consider
        min_samples_leaf: Minimum samples required in each leaf
    
    Returns:
        Tuple of (best_feature, best_split_value)
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert len(X) == len(y), "X and y must have the same number of samples"
    assert criterion in ['entropy', 'gini', 'mse'], f"Unknown criterion: {criterion}"
    
    best_attr = None
    best_val = None
    best_gain = -np.inf

    for attr in features:
        if attr not in X.columns:
            continue
            
        col = X[attr]
        
        # Skip if all values are missing
        if col.isna().all():
            continue

        # Numeric case
        if pd.api.types.is_numeric_dtype(col):
            # Get unique non-null values and sort them
            unique_vals = sorted(col.dropna().unique())
            
            if len(unique_vals) <= 1:
                continue
            
            # Try threshold splits between consecutive unique values
            for i in range(len(unique_vals) - 1):
                threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
                
                # Create masks for split
                left_mask = col < threshold
                right_mask = col >= threshold
                
                # Check minimum samples constraint
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue
                
                # Calculate information gain for this split
                gain = calculate_split_gain(y, left_mask, criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_val = threshold

        # Categorical case
        else:
            unique_vals = col.dropna().unique()
            
            # Try binary splits: each value vs rest
            for val in unique_vals:
                left_mask = col == val
                right_mask = col != val
                
                # Check minimum samples constraint
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue
                
                # Calculate information gain for this split
                gain = calculate_split_gain(y, left_mask, criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_val = val

    return best_attr, best_val

def calculate_split_gain(y: pd.Series, left_mask: pd.Series, criterion: str) -> float:
    """
    Calculate information gain for a binary split defined by left_mask
    """
    if y.empty or left_mask.empty:
        return 0.0
    
    right_mask = ~left_mask
    n_total = len(y)
    n_left = left_mask.sum()
    n_right = right_mask.sum()
    
    # Check for empty splits
    if n_left == 0 or n_right == 0:
        return 0.0
    
    # Calculate parent and child impurities
    parent_impurity = compute_impurity(y, criterion)
    left_impurity = compute_impurity(y[left_mask], criterion)
    right_impurity = compute_impurity(y[right_mask], criterion)
    
    # Calculate weighted child impurity
    weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
    
    # Information gain
    gain = parent_impurity - weighted_impurity
    return max(0.0, gain)

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, 
               value: Union[float, str, None]) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Function to split the data according to an attribute and value.
    
    Args:
        X: Input features DataFrame
        y: Target variable Series
        attribute: Feature name to split on
        value: Split value/threshold
    
    Returns:
        Tuple of ((X_left, y_left), (X_right, y_right))
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert attribute in X.columns, f"Attribute '{attribute}' not found in X"
    assert len(X) == len(y), "X and y must have the same number of samples"
    
    if value is None:
        # Return original data in left split if no value provided
        return (X.copy(), y.copy()), (pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype))
    
    col = X[attribute]
    
    # Handle different data types
    if pd.api.types.is_numeric_dtype(col):
        # Numeric split: left < threshold, right >= threshold
        left_mask = col < value
        right_mask = col >= value
    else:
        # Categorical split: left == value, right != value
        left_mask = col == value
        right_mask = col != value

    # Apply masks to split data
    X_left = X[left_mask].copy()
    y_left = y[left_mask].copy()
    X_right = X[right_mask].copy()
    y_right = y[right_mask].copy()

    return (X_left, y_left), (X_right, y_right)

def is_pure(Y: pd.Series) -> bool:
    """
    Check if all values in Y are the same (pure node)
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    
    if Y.empty:
        return True
    
    return Y.nunique(dropna=False) <= 1

def leaf_value(Y: pd.Series) -> Union[float, str, int, None]:
    """
    Determine the prediction value for a leaf node
    
    For classification: return the most frequent class
    For regression: return the mean value
    """
    assert isinstance(Y, pd.Series), "Y must be a pandas Series"
    
    if Y.empty:
        return None
    
    if check_ifreal(Y):
        # Regression: return mean
        return float(Y.mean())
    else:
        # Classification: return most frequent class
        mode_values = Y.mode()
        if not mode_values.empty:
            return mode_values.iloc[0]
        else:
            return Y.iloc[0]

def validate_input(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validate input data for decision tree
    """
    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y must be a pandas Series"
    assert len(X) == len(y), "X and y must have the same number of samples"
    assert len(X) > 0, "Input data cannot be empty"
    assert len(X.columns) > 0, "X must have at least one feature"

def get_feature_types(X: pd.DataFrame) -> dict:
    """
    Get the data types of features
    """
    feature_types = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'categorical'
    return feature_types

def handle_missing_values(X: pd.DataFrame, strategy: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        X: Input DataFrame
        strategy: 'mode' for categorical, 'mean' for numeric, 'drop' to remove rows
    
    Returns:
        DataFrame with missing values handled
    """
    X_processed = X.copy()
    
    if strategy == 'drop':
        X_processed = X_processed.dropna()
    elif strategy == 'mode':
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                else:
                    mode_val = X_processed[col].mode()
                    if not mode_val.empty:
                        X_processed[col].fillna(mode_val.iloc[0], inplace=True)
    
    return X_processed