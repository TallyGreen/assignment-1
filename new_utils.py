import pickle
import numpy as np
from sklearn.metrics import accuracy_score

from collections import Counter
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from typing import Type, Dict
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate, StratifiedKFold
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import (
    cross_validate,
    KFold,
)
"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
# new_utils.py

# new_utils.py

def scale_data(X):
    X = X.astype(float)
    X = X / X.max()
    return X

def load_mnist_dataset(
    nb_samples=None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Load the MNIST dataset.

    nb_samples: number of samples to save. Useful for code testing.
    The homework requires you to use the full dataset.

    Returns:
        X, y
        #X_train, y_train, X_test, y_test
    """

    try:
        # Are the datasets already loaded?
        print("... Is MNIST dataset local?")
        X: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except Exception as e:
        # Download the datasets
        print(f"load_mnist_dataset, exception {e}, Download file")
        X, y = datasets.fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        X = X.astype(float)
        y = y.astype(int)

    y = y.astype(np.int32)
    X: NDArray[np.floating] = X
    y: NDArray[np.int32] = y

    if nb_samples is not None and nb_samples < X.shape[0]:
        X = X[0:nb_samples, :]
        y = y[0:nb_samples]

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", X)
    np.save("mnist_y.npy", y)
    return X, y


def prepare_data(num_train: int = 60000, num_test: int = 10000, normalize: bool = True):
    """
    Prepare the data.
    Parameters:
        X: A data matrix
        frac_train: Fraction of the data used for training in [0,1]
    Returns:
    Prepared data matrix
    Side effect: update of self.Xtrain, self.ytrain, self.Xtest, self.ytest
    """
    # Check in case the data is already on the computer.
    X, y = load_mnist_dataset()

    # won't work well unless X is greater or equal to zero
    if normalize:
        X = X / X.max()

    y = y.astype(np.int32)
    Xtrain, Xtest = X[:num_train], X[num_train : num_train + num_test]
    ytrain, ytest = y[:num_train], y[num_train : num_train + num_test]
    return Xtrain, ytrain, Xtest, ytest

    
