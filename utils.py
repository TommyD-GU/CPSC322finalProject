import copy
import numpy as np
from tabulate import tabulate


def euclid_dist(pt1, pt2):
    """Computes the Euclidean distance between two points.

    Args:
        pt1 (list): First point as a list of coordinates.
        pt2 (list): Second point as a list of coordinates.

    Returns:
        float: The Euclidean distance between pt1 and pt2.
    """
    # Calculate the sum of squared differences
    sum_squared_diff = sum((x1 - x2) ** 2 for x1, x2 in zip(pt1, pt2))
    return sum_squared_diff ** 0.5  # Return the square root of the sum of squared differences


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Set a specific seed for random number generation if a random state is provided
    """Split dataset into train and test sets based on a test set size."""
    # Set a specific seed for random number generation if a random state is provided
    if random_state is not None:
        np.random.seed(random_state)

    # Create deep copies of X and y to avoid modifying the original data
    X_copy = copy.deepcopy(X)
    y_copy = copy.deepcopy(y)

    # Shuffle the data if the shuffle flag is set to True
    if shuffle:
        randomize_in_place(X_copy, y_copy)  # Removed random_state here

    # Get the total number of samples
    n_samples = len(X)

    # Convert test_size from a float to an integer if it's given as a proportion
    if isinstance(test_size, float):
        test_size = int((n_samples * test_size) + 0.999)  # Adding 0.999 ensures rounding up

    # Check that test_size is smaller than the number of samples; raise an error otherwise
    if test_size >= n_samples:
        raise ValueError("Test size must be smaller than the total number of samples")

    # Calculate the split index where the data should be divided into training and testing sets
    split_index = n_samples - test_size

    # Split the X and y copies into training and testing sets
    X_train = X_copy[:split_index]
    X_test = X_copy[split_index:]
    y_train = y_copy[:split_index]
    y_test = y_copy[split_index:]

    # Return the split data: training and testing sets for both features and labels
    return X_train, X_test, y_train, y_test

def randomize_in_place(alist, parallel_list=None):
    '''shuffle in place

    args: a list and optional second list

    returns the list(s) in shuffled order
    '''

    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def normalize_data(data):
    """Normalizes the input list of data."""
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]