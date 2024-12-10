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
import numpy as np

import random
import os
from collections import Counter
import math

import copy
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
                distance = utils.euclid_dist(test_instance, train_instance)  # Calculate distance
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

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # Count occurrences of each label in y_train to compute priors
        label_counts = {}
        for label in y_train:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Calculate priors
        total_samples = len(y_train)
        self.priors = {label: count / total_samples for label, count in label_counts.items()}

        # Initialize posteriors
        self.posteriors = {label: [{} for _ in range(len(X_train[0]))] for label in label_counts}

        # Count occurrences of each attribute value given each label
        for i in range(len(X_train)):
            label = y_train[i]
            for j, value in enumerate(X_train[i]):
                if value not in self.posteriors[label][j]:
                    self.posteriors[label][j][value] = 1
                else:
                    self.posteriors[label][j][value] += 1

        # Calculate posteriors
        for label, feature_counts in self.posteriors.items():
            for j, value_counts in enumerate(feature_counts):
                total_label_count = label_counts[label]
                self.posteriors[label][j] = {
                    value: count / total_label_count for value, count in value_counts.items()
                }

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            label_probs = {}
            for label in self.priors:
                # Start with the prior probability of the label
                label_prob = self.priors[label]
                # Multiply by each attribute's conditional probability given the label
                for j, value in enumerate(instance):
                    # Use a small probability if the value hasn't been seen in training data
                    label_prob *= self.posteriors[label][j].get(value, 1e-6)
                label_probs[label] = label_prob

            # Predict the label with the highest probability
            best_label = max(label_probs, key=label_probs.get)
            y_predicted.append(best_label)

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # Generate default attribute names
        attribute_names = [f"att{i}" for i in range(len(X_train[0]))]

        # Combine training data with labels
        training_data = [row + [label] for row, label in zip(X_train, y_train)]

        # Use external TDIDT function to build the tree
        self.tree = utils.tdidt(training_data, attribute_names, len(training_data))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predict = []
        for instance in X_test:
            tree = self.tree
            while tree[0] != "Leaf":
                attribute_index = int(tree[1][3:])
                instance_value = instance[attribute_index]

                found_branch = False
                for i in range(2, len(tree)):
                    if tree[i][1] == instance_value:
                        tree = tree[i][2]
                        found_branch = True
                        break

                if not found_branch:
                # Randomly select a branch if no matching branch is found
                    random_branch = random.choice(tree[2:])
                    tree = random_branch[2]
            y_predict.append(tree[1])
        return y_predict

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        element = [(self.tree, [])]  # Each element is (current_node, rule_conditions)

        while element:
            current_node, rule_conditions = element.pop()

            if current_node[0] == "Leaf":
                # Leaf node: Print the rule
                rule = "IF " + " AND ".join(rule_conditions)
                rule += f" THEN {class_name} = {current_node[1]}"
                print(rule)
            else:
                # Attribute node
                attribute = current_node[1]
                attribute_name = (
                    attribute_names[int(attribute[3:])] if attribute_names else attribute
                )

                for i in range(2, len(current_node)):
                    value = current_node[i][1]
                    # Add new rule condition and push subtree to element
                    element.append((current_node[i][2], rule_conditions + [f"{attribute_name} == {value}"]))

    # # BONUS method
    # def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
    #     """
    #     Visualizes a decision tree using Graphviz and generates a .dot and .pdf file.

    #     Args:
    #         dot_fname (str): The name of the .dot output file.
    #         pdf_fname (str): The name of the .pdf output file generated from the .dot file.
    #         attribute_names (list of str or None): A list of attribute names to use in the visualization
    #             (None if a list is not provided, in which case default attribute names based on
    #             indexes (e.g., "att0", "att1", ...) will be used).
    #     """
    #     def add_nodes_edges(dot, tree, parent_name=None):
    #         """Recursively adds nodes and edges to the Graphviz Digraph."""
    #         if tree[0] == "Leaf":
    #             # Leaf node
    #             node_name = f'leaf_{id(tree)}'  # Unique identifier for each node
    #             label = f"Class = {tree[1]}\n({tree[2]}/{tree[3]})"
    #             dot.node(node_name, label=label, shape="ellipse", style="filled", color="lightblue")
    #             if parent_name:
    #                 dot.edge(parent_name, node_name)
    #         else:
    #             # Attribute node
    #             attribute_name = (
    #                 attribute_names[int(tree[1][3:])] if attribute_names else tree[1]
    #             )
    #             node_name = f'node_{id(tree)}'  # Unique identifier for each node
    #             dot.node(node_name, label=attribute_name, shape="box", style="rounded,filled", color="lightyellow")
    #             if parent_name:
    #                 dot.edge(parent_name, node_name)

    #             # Recursively process children
    #             for i in range(2, len(tree)):
    #                 value = tree[i][1]
    #                 child_name = f'{node_name}_{value}'  # Unique edge name
    #                 add_nodes_edges(dot, tree[i][2], node_name)

    #     # Create a Graphviz Digraph
    #     dot = Digraph()

    #     # Add nodes and edges recursively
    #     add_nodes_edges(dot, self.tree)

    #     # Save the .dot file and render it as a .pdf
    #     dot.render(filename=dot_fname, format="pdf", cleanup=True)

    #     # Ensure the generated PDF is saved in the required directory
    #     os.makedirs("tree_vis", exist_ok=True)
    #     os.rename(f"{dot_fname}.pdf", f"tree_vis/{pdf_fname}.pdf")


class MyRandomForestClassifier:
    """Random Forest implementation using MyDecisionTreeClassifier."""

    def __init__(self, n_trees=10, max_features=None, sample_size=0.8, max_ensemble_trees=None):
        """
        Initialize the Random Forest.
        Args:
            n_trees (int): Total number of trees (N).
            max_features (int): Number of attributes to consider at each split (F).
            sample_size (float): Fraction of the dataset to use for each tree.
            max_ensemble_trees (int): Number of best trees to use in the final ensemble (M).
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.sample_size = sample_size
        self.max_ensemble_trees = max_ensemble_trees
        self.trees = []

    def fit(self, X_train, y_train):
        """Fit the Random Forest model."""
        n_samples = len(X_train)
        n_features = len(X_train[0])
        self.max_features = self.max_features or int(n_features ** 0.5)

        for _ in range(self.n_trees):
            # Bootstrap sampling
            sample_indices = random.choices(range(n_samples), k=int(n_samples * self.sample_size))
            X_sample = [X_train[i] for i in sample_indices]
            y_sample = [y_train[i] for i in sample_indices]

            # Create and fit a decision tree
            tree = MyDecisionTreeClassifier()
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, sample_indices))

        # Select the best subset of trees if M < N
        if self.max_ensemble_trees and self.max_ensemble_trees < self.n_trees:
            self._select_best_trees(X_train, y_train)

    def _select_best_trees(self, X_train, y_train):
        """Select the best subset of trees based on training accuracy."""
        tree_accuracies = []
        for tree, sample_indices in self.trees:
            X_sample = [X_train[i] for i in sample_indices]
            y_sample = [y_train[i] for i in sample_indices]
            predictions = tree.predict(X_sample)
            accuracy = sum(1 for y_true, y_pred in zip(y_sample, predictions) if y_true == y_pred) / len(y_sample)
            tree_accuracies.append((tree, accuracy))

        # Sort by accuracy and keep the top M trees
        tree_accuracies.sort(key=lambda x: -x[1])
        self.trees = [tree for tree, _ in tree_accuracies[:self.max_ensemble_trees]]

    def predict(self, X_test):
        """Predict the class labels for the input data."""
        predictions = []
        for instance in X_test:
            votes = {}
            for tree, _ in self.trees:
                prediction = tree.predict([instance])[0]
                votes[prediction] = votes.get(prediction, 0) + 1

            # Majority voting
            predictions.append(max(votes, key=votes.get))
        return predictions