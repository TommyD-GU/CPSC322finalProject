"""
##########################################################################
# Programmer: Malia Recker
# Class: CPSC 322 Fall 2024
# Programming Assignment #6
# 11/10/2024
#
# Description:
# Functions for data splitting, cross-validation, and evaluation metrics,
# including train-test split, k-fold and stratified k-fold splitting,
# bootstrapping, confusion matrix computation, and accuracy scoring.
#
############################################ ###############################
"""
import copy
import numpy as np # use numpy's random number generation
import utils
# from mysklearn import utils
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
    if random_state is not None:
        np.random.seed(random_state)

    # Create deep copies of X and y to avoid modifying the original data
    X_copy = copy.deepcopy(X)
    y_copy = copy.deepcopy(y)

    # Shuffle the data if the shuffle flag is set to True
    if shuffle:
        utils.randomize_in_place(X_copy, y_copy)  # Removed random_state here

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

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Number of samples
    n_samples = len(y)

    # Create a list of indices for the samples
    indices = np.arange(n_samples)

    # Shuffle the indices if specified
    if shuffle:
        utils.randomize_in_place(indices, seed=random_state)

    # Get unique classes and their counts
    unique_classes, _ = np.unique(y, return_counts=True)

    # Prepare folds
    folds = []

    # Create a list to hold the indices for each class
    class_indices = {cls: [] for cls in unique_classes}

    # Populate the class indices
    for index in indices:
        class_indices[y[index]].append(index)

    # Calculate the number of samples per fold for each class
    samples_per_fold = {cls: len(class_indices[cls]) // n_splits for cls in unique_classes}

    # Create the stratified folds
    for fold in range(n_splits):
        train_indices = []
        test_indices = []

        # Create test set for this fold
        for cls in unique_classes:
            # Determine the start and end indices for the test samples
            start_index = fold * samples_per_fold[cls]
            if fold == n_splits - 1:  # Last fold may take the remainder
                end_index = len(class_indices[cls])
            else:
                end_index = start_index + samples_per_fold[cls]

            # Select test indices
            test_indices.extend(class_indices[cls][start_index:end_index])

            # Add remaining indices to the training set
            train_indices.extend(class_indices[cls][:start_index])
            train_indices.extend(class_indices[cls][end_index:])

        # Append the fold as a tuple of training and testing indices
        folds.append((train_indices, test_indices))

    return folds