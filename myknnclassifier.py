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
import utils
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        all_distances = []
        all_neighbors = []
        
        for test_instance in X_test:
            dist = []
            
            # do euclidean distance
            for i, train_instance in enumerate(self.X_train):
                distance = utils.euclid_dist(test_instance, train_instance)
                dist.append((distance, i))
                
            # sort the dist
            for j in range(len(dist)):
                for k in range(j + 1, len(dist)):
                    if dist[j][0] > dist[k][0]:
                        dist[j], dist[k] = dist[k], dist[j]
            
            # take the k smallest distances and their indices
            current_distances = [d[0] for d in dist[:self.n_neighbors]]
            current_neighbors = [d[1] for d in dist[:self.n_neighbors]]
            
            all_distances.append(current_distances)
            all_neighbors.append(current_neighbors)
        
        return all_distances, all_neighbors
                

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        # get the nearest neighbors using kneighbors
        _, neighbor_indices = self.kneighbors(X_test)

        y_predicted = []

        # for each test instance, get the labels of its nearest neighbors
        for neighbors in neighbor_indices:
            # get the corresponding labels from y_train
            neighbor_labels = [self.y_train[i] for i in neighbors]

            # use the most frequent label of the neighbors
            most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)

            y_predicted.append(most_common_label)

        return y_predicted
    

def mixed_distance(instance1, instance2, categorical_indices):
    """Calculates the mixed distance between two instances.

    Args:
        instance1 (list): First instance.
        instance2 (list): Second instance.
        categorical_indices (list of int): List of indices for categorical features.

    Returns:
        float: The mixed distance between instance1 and instance2.
    """
    distance = 0
    for i, (val1, val2) in enumerate(zip(instance1, instance2)):
        if i in categorical_indices:
            # Categorical: add 1 if values are different, 0 if they are the same
            distance += 0 if val1 == val2 else 1
        else:
            # Numerical: use Euclidean distance component
            distance += (val1 - val2) ** 2
    return distance ** 0.5  # Take square root for Euclidean component

class MyKNeighborsCatClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical_indices=None):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
            categorical_indices(list of int, optional): List of indices of categorical features
        """
        self.n_neighbors = n_neighbors
        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        all_distances = []
        all_neighbors = []

        for test_instance in X_test:
            dist = []
            for i, train_instance in enumerate(self.X_train):
                # Use mixed distance function
                distance = mixed_distance(test_instance, train_instance, self.categorical_indices)
                dist.append((distance, i))
            dist.sort(key=lambda x: x[0])  # Sort based on distance
            current_distances = [d[0] for d in dist[:self.n_neighbors]]
            current_neighbors = [d[1] for d in dist[:self.n_neighbors]]
            all_distances.append(current_distances)
            all_neighbors.append(current_neighbors)

        return all_distances, all_neighbors

    def predict(self, X_test):
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for neighbors in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in neighbors]
            most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
            y_predicted.append(most_common_label)
        return y_predicted
