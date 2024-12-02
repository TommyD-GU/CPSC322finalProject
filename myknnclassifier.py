"""
###########################################################################
# Programmer: Malia Recker and Tommy Dunne
# Class: CPSC 322 Fall 2024
# Final Project
# 10/18/2024
#
# Description: This module contains reusable general-purpose functions
# designed for various data processing tasks. The functions include
# normalization of numerical data, calculation of Euclidean distances,
# discretization of values based on specified criteria, generation of
# random indices for test data selection, and splitting datasets into
# training and test sets. These functions facilitate the handling and
# manipulation of datasets, particularly for machine learning tasks.
#
###########################################################################
"""
class MyKNeighborsClassifier:
    """Represents a simple k-nearest neighbors classifier.

    Attributes:
        n_neighbors (int): Number of k neighbors.
        X_train (list of list of numeric vals): The list of training instances (samples).
            The shape of X_train is (n_train_samples, n_features).
        y_train (list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples.

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column.
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors (int): Number of k neighbors.
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train (list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features).
            y_train (list of obj): The target y values (parallel to X_train).
                The shape of y_train is n_train_samples.

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train.
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test (list of list of numeric vals): The list of testing samples.
                The shape of X_test is (n_test_samples, n_features).

        Returns:
            distances (list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test.
            neighbor_indices (list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances).
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            dist = []
            # Compute Euclidean distance
            for i, train_instance in enumerate(self.X_train):
                distance = euclid_dist(test_instance, train_instance)  # Calculate distance
                dist.append((distance, i))  # Store distance and index
            dist.sort(key=lambda x: x[0])  # Sort by distance
            # Extract the k smallest distances and their corresponding indices
            k_distances = [d[0] for d in dist[:self.n_neighbors]]
            k_indices = [d[1] for d in dist[:self.n_neighbors]]
            distances.append(k_distances)
            neighbor_indices.append(k_indices)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of numeric vals): The list of testing samples.
                The shape of X_test is (n_test_samples, n_features).

        Returns:
            y_predicted (list of obj): The predicted target y values (parallel to X_test).
        """
        _, neighbor_indices = self.kneighbors(X_test)  # Get neighbor indices
        y_predicted = []
        for indices in neighbor_indices:  # Predict based on majority vote
            neighbor_classes = [self.y_train[i] for i in indices]  # Get classes of neighbors
            predicted_class = max(set(neighbor_classes), key=neighbor_classes.count)  # Majority vote
            y_predicted.append(predicted_class)  # Append predicted class
        return y_predicted

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
