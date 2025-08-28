from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    assert y.size > 0 
    return (y_hat==y).sum()/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positive = (y_hat == cls) & (y == cls)
    false_positive = (y_hat == cls) & (y!=cls)
    if (true_positive.sum() + false_positive.sum()) == 0:
        return 0.0
    return true_positive.sum() / (true_positive.sum() + false_positive.sum())


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positive = (y_hat == cls) & (y == cls)
    false_negative = (y_hat !=cls) & (y == cls)
    if (true_positive.sum() + false_negative.sum()) == 0:
        return 0.0
    return true_positive.sum() / (true_positive.sum() + false_negative.sum())

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    return ((y_hat - y) ** 2).mean() ** 0.5



def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return (y_hat - y).abs().mean()
