import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from mypytable import MyPyTable



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

    return normalized_data

def rand_inds(num, total):
    """Creates a list of num random indices from 0 to total-1."""
    random.seed(1)
    inds = []
    for _ in range(num):  # Iterate num times
        inds.append(random.randint(0, total - 1))  # Adjusted to valid index range
    return inds

def freq_plot(data, header,col_name_x, lables=None):
    """Generates a frequency plot for the specified column in the data file.

    Args:
        infile (str): The name of the input file containing the data.
        col_name_x (str): The name of the column to plot frequencies for.

    Returns:
        None
    """
    # Load the data table from the specified file
    tbl = MyPyTable(header, data)
    x_unsorted = tbl.get_column(col_name_x)
    x_list = sorted(x_unsorted)
    unique_vals = []  # To hold unique values
    y = []  # To hold frequency counts

    # Loop through each sorted value to count frequencies
    for value in x_list:
        if value in unique_vals:
            # Increment frequency count for existing value
            index = unique_vals.index(value)
            y[index] += 1
        else:
            # Add new value and initialize its frequency count
            unique_vals.append(value)
            y.append(1)

    x = set(x_list)  # Extract unique values
    x = list(x)
    # Display x and y values
    print("x values:", x, "y values:", y)

    # Create a bar plot
    plt.figure()
    color1= generate_cute_color()
    plt.bar(x, y, color=color1)
    # Label plot
    plt.xlabel(col_name_x)
    plt.ylabel('count')
    plt.title(f'Frequency plot of {col_name_x}')
    plt.show()



def generate_cute_color():
    """
    Generates a random hex color within a "cute" range.

    Returns:
        str: A hex color string (e.g., "#FFB6C1").
    """
    # Define ranges for "cute" colors (lighter tones)
    # Higher red, green, and blue values result in pastel-like colors
    red = random.randint(180, 255)
    green = random.randint(150, 255)
    blue = random.randint(200, 255)

    # Convert to hex
    return f"#{red:02X}{green:02X}{blue:02X}"


def calculate_accuracy_error_rate(results):
    """Calculate the accuracy and error rate from the results of cross-validation.

    Args:
        results (list of tuples): Each tuple contains the true labels (y_test) and
            the predicted labels (y_pred). Each tuple is of the form (y_test, y_pred).

    Returns:
        accuracy (float): The accuracy of the classifier.
        error_rate (float): The error rate of the classifier.
    """
    correct = 0  # Initialize a counter for correct predictions
    total = 0    # Initialize a counter for the total number of predictions

    # Iterate through the results for each fold
    for y_test, y_pred in results:
        # Count the number of correct predictions for each fold
        correct += sum([1 for true, pred in zip(y_test, y_pred) if true == pred])
        total += len(y_test)  # Update the total number of predictions

    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy = correct / total
    # Calculate error rate as the complement of accuracy
    error_rate = 1 - accuracy

    return accuracy, error_rate

def calculate_precision_recall_f1(results):
    """Calculate precision, recall, and F1 score from the results of cross-validation.

    Args:
        results (list of tuples): Each tuple contains the true labels (y_test)
            and the predicted labels (y_pred). Each tuple is of the form (y_test, y_pred).

    Returns:
        precision (float): The precision of the classifier.
        recall (float): The recall of the classifier.
        f1 (float): The F1 score of the classifier.
    """
    true_positive = 0  # Initialize counter for true positives
    false_positive = 0  # Initialize counter for false positives
    false_negative = 0  # Initialize counter for false negatives
    true_negative = 0  # Initialize counter for true negatives

    # Iterate through each fold's true and predicted values
    for y_test, y_pred in results:
        for true, pred in zip(y_test, y_pred):
            # Count true positives, false positives, false negatives, and true negatives
            if true == 1 and pred == 1:
                true_positive += 1
            elif true == 0 and pred == 1:
                false_positive += 1
            elif true == 1 and pred == 0:
                false_negative += 1
            elif true == 0 and pred == 0:
                true_negative += 1

    # Calculate precision (how many predicted positives are actually positive)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    # Calculate recall (how many actual positives were correctly identified)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    # Calculate F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def calculate_confusion_matrix(results):
    """Calculate the confusion matrix from the results of cross-validation.

    Args:
        results (list of tuples): Each tuple contains the true labels (y_test)
            and the predicted labels (y_pred). Each tuple is of the form (y_test, y_pred).

    Returns:
        conf_matrix (list of list): A confusion matrix as a 2x2 list of counts.
            The matrix will have the following format:
            [[true_negative, false_positive],
             [false_negative, true_positive]]
    """
    true_positive = 0  # Initialize counter for true positives
    false_positive = 0  # Initialize counter for false positives
    false_negative = 0  # Initialize counter for false negatives
    true_negative = 0  # Initialize counter for true negatives

    # Iterate through each fold's true and predicted values
    for y_test, y_pred in results:
        for true, pred in zip(y_test, y_pred):
            # Count true positives, false positives, false negatives, and true negatives
            if true == 1 and pred == 1:
                true_positive += 1
            elif true == 0 and pred == 1:
                false_positive += 1
            elif true == 1 and pred == 0:
                false_negative += 1
            elif true == 0 and pred == 0:
                true_negative += 1

    # Create confusion matrix in the format: [[TN, FP], [FN, TP]]
    conf_matrix = [[true_negative, false_positive],
                   [false_negative, true_positive]]

    return conf_matrix

def compute_accuracy(y_pred, y_test):
    """
    Computes the accuracy of predictions.

    Args:
        y_pred (list): List of predicted values.
        y_test (list): List of true values.

    Returns:
        float: Accuracy as a proportion (between 0 and 1).
    """
    if len(y_pred) != len(y_test):
        raise ValueError("The lengths of y_pred and y_test must be the same.")

    # Count the number of correct predictions
    correct_predictions = sum(1 for pred, true in zip(y_pred, y_test) if pred == true)

    # Compute accuracy as the ratio of correct predictions to the total number of instances
    accuracy = correct_predictions / len(y_test)
    return accuracy

def compute_recall(y_pred, y_test, positive_label=1):
    """
    Computes the recall of predictions.

    Args:
        y_pred (list): List of predicted values.
        y_test (list): List of true values.
        positive_label (obj): The label considered as the positive class (default is 1).

    Returns:
        float: Recall as a proportion (between 0 and 1).
    """
    if len(y_pred) != len(y_test):
        raise ValueError("The lengths of y_pred and y_test must be the same.")

    # Count the number of true positives and total actual positives
    true_positives = sum(1 for pred, true in zip(y_pred, y_test) if pred == true == positive_label)
    actual_positives = sum(1 for true in y_test if true == positive_label)

    # Handle case where there are no positive instances in y_test
    if actual_positives == 0:
        return 0.0  # Recall is undefined but often set to 0 when there are no positives

    # Compute recall
    recall = true_positives / actual_positives
    return recall

def randomize_in_place(alist, parallel_list=None, seed=0):
    """
    Randomizes the order of elements in `alist` in place by swapping each element with another random element.
    If a `parallel_list` is provided, it will be shuffled in the same order to maintain alignment with `alist`.

    Args:
        alist (list): The main list to be shuffled in place.
        parallel_list (list, optional): A second list to shuffle in parallel with `alist`. The shuffle
            will maintain the correspondence of elements between `alist` and `parallel_list`.
        seed (int): Seed value for the random number generator to ensure reproducibility (default is 0).

    Note:
        This function is part of U4-Supervised-Learning/ClassificationFun/main.py, written by Gina Sprint.
    """
    np.random.seed(seed)
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
