from mysklearn.mydecisiontreeclassifier import MyDecisionTreeClassifier

import random
import copy
import numpy as np
from collections import Counter
import utils

class MyRandomForestClassifier:
    def __init__(self, N=20, M=7, F=2):
        """
        Initializes the random forest classifier with given parameters.
        Args:
            N (int): Number of trees to generate.
            M (int): Number of trees to use in voting for predictions.
            F (int): Number of features to consider for splitting at each node.
        """
        self.N = N
        self.M = M
        self.F = F
        self.trees = []

    def bootstrap_sample(self, X, y=None, n_samples=None, random_state=None):
        """Split dataset into bootstrapped training set and out of bag test set.

        Args:
            X(list of list of obj): The list of samples
            y(list of obj): The target y values (parallel to X)
                Default is None (in this case, the calling code only wants to sample X)
            n_samples(int): Number of samples to generate. If left to None (default) this is automatically
                set to the first dimension of X.
            random_state(int): integer used for seeding a random number generator for reproducible results

        Returns:
            X_sample(list of list of obj): The list of samples
            X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
            y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
                None if y is None
            y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
                None if y is None
        Notes:
            Loosely based on sklearn's resample():
                https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
            Sample indexes of X with replacement, then build X_sample and X_out_of_bag
                as lists of instances using sampled indexes (use same indexes to build
                y_sample and y_out_of_bag)
        """
        # Set random seed if provided for reproducibility
        if random_state is None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng(1)

        # Determine number of samples to generate
        if n_samples is None:
            n_samples = len(X)

        # Generate indices for bootstrapping
        indices = rng.integers(low=0, high=len(X), size=n_samples)  # Sample with replacement

        X_sample = [X[i] for i in indices]  # Bootstrapped sample
        y_sample = [y[i] for i in indices] if y is not None else None  # Bootstrapped target if y is provided

        # Determine out-of-bag indices
        out_of_bag_indices = list(set(range(len(X))) - set(indices))  # Indices not sampled
        X_out_of_bag = [X[i] for i in out_of_bag_indices]  # Out-of-bag samples
        y_out_of_bag = [y[i] for i in out_of_bag_indices] if y is not None else None  # Out-of-bag targets if y is provided

        return X_sample, y_sample, X_out_of_bag, y_out_of_bag


    def select_random_features(self, features):
        """
        Selects a random subset of features for splitting at each node.
        Args:
            features (list): List of all available features.
        Returns:
            list: A random subset of features of size F.
        """
        return random.sample(features, self.F)

    def stratified_split(self, X, y, test_size=0.33, random_state=None, shuffle=True):
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
        if random_state is not None:
            np.random.seed(random_state)
        else:
            np.random.seed(1)

        # Create deep copies of X and y to avoid modifying the original data
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)

        # Shuffle the data if the shuffle flag is set to True
        if shuffle:
            utils.randomize_in_place(X_copy, y_copy, random_state)

        # Get the total number of samples
        n_samples = len(X)

        # Convert test_size from a float to an integer if it's given as a proportion
        if isinstance(test_size, float):
            test_size = int((n_samples * test_size) + 0.999)    # Adding 0.999 ensures rounding up

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

    def fit(self, X, y):
        """
        Fits the random forest model using N decision trees.
        Args:
            X (list of list): The feature matrix.
            y (list): The target values.
        """
        # Split the data into train/validation sets using stratified split
        # X_train, y_train, X_val, y_val = self.stratified_split(X, y)

        self.trees = []
        all_features = list(range(len(X[0])))  # List of feature indices

        # Generate N decision trees
        for _ in range(self.N):
            # Create a bootstrapped sample
            X_sample, y_sample, X_val, y_val = self.bootstrap_sample(X,y)

            # Create a random decision tree using the bootstrapped sample and random features
            tree = MyDecisionTreeClassifier()
            utils.print_tree(tree)
            tree.fit(X_sample, y_sample)

            # Add the tree to the forest
            self.trees.append(tree)

        # After the trees are generated, evaluate them on the validation set and select the M best ones
        tree_accuracies = []

        for tree in self.trees:
            # Get the accuracy of the tree on the validation set
            predictions = [tree.tdidt_predict(tree.tree, instance, tree.header) for instance in X_val]
            accuracy = sum([1 for pred, true in zip(predictions, y_val) if pred == true]) / len(y_val)
            tree_accuracies.append((tree, accuracy))

        # Sort trees by accuracy and select the M most accurate
        tree_accuracies.sort(key=lambda x: x[1], reverse=True)
        self.trees = [tree for tree, _ in tree_accuracies[:self.M]]

    def predict_one(self, instance):
        """
        Predict the class label for a single instance using majority voting from M trees.
        Args:
            instance (list): The instance to classify.
        Returns:
            The predicted class label.
        """
        # Gather predictions from the M most accurate trees
        predictions = [tree.tdidt_predict(tree.tree, instance, tree.header) for tree in self.trees]

        # Majority voting
        majority_vote = Counter(predictions).most_common(1)[0][0]
        return majority_vote

    def predict(self, X_test):
        """
        Predict the class labels for a list of test instances.
        Args:
            X_test (list of list): The test feature matrix.
        Returns:
            list: The predicted class labels for each instance in the test set.
        """
        return [self.predict_one(instance) for instance in X_test]

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data by computing the accuracy.
        Args:
            X_test (list of list): The test feature matrix.
            y_test (list): The true labels for the test set.
        Returns:
            float: The accuracy of the model on the test data.
        """
        predictions = self.predict(X_test)
        correct = sum([1 for pred, true in zip(predictions, y_test) if pred == true])
        accuracy = correct / len(y_test)
        return accuracy
