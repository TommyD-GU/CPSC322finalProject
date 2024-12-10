
import random
import matplotlib.pyplot as plt
import numpy as np
from mysklearn.mypytable import MyPyTable
import math

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

def sample_classification_data(dataset, classification_index=8, sample_size=1000):
    """
    Randomly samples 1000 rows for each classification (1 and 0) from the dataset.

    Parameters:
    - dataset: List of lists, where each inner list represents a row in the dataset.
    - classification_index: The index of the classification attribute in each row (default is 8).
    - sample_size: Number of rows to sample for each classification (default is 1000).

    Returns:
    - sampled_data: A new dataset containing the sampled rows for each classification.
    """
    # Separate rows based on their classification
    class_0 = [row for row in dataset if row[classification_index] == 0]
    class_1 = [row for row in dataset if row[classification_index] == 1]

    # Ensure there are enough rows to sample from
    if len(class_0) < sample_size or len(class_1) < sample_size:
        raise ValueError("Not enough rows in one or both classifications to sample the requested size.")

    # Randomly sample 1000 rows from each classification
    sampled_class_0 = random.sample(class_0, sample_size)
    sampled_class_1 = random.sample(class_1, sample_size)

    # Combine the sampled rows and shuffle them
    sampled_data = sampled_class_0 + sampled_class_1
    random.shuffle(sampled_data)

    return sampled_data

def plot_scatter(data, header, col_name_x, col_name_y=None):
    """
    Plots a scatter plot for the specified column in the data.

    Args:
        data (list of lists): The data table as a list of rows.
        header (list of str): The column headers for the data.
        col_name_x (str): The name of the column for the x-axis.
        col_name_y (str): The name of the column for the y-axis (optional).
                          If None, uses row indices as y values.

    Returns:
        None
    """
    # Find column indices
    x_index = header.index(col_name_x)
    x_values = [row[x_index] for row in data]

    # If col_name_y is provided, use it; otherwise, use indices as y-values
    if col_name_y:
        y_index = header.index(col_name_y)
        y_values = [row[y_index] for row in data]
    else:
        y_values = list(range(len(data)))

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', alpha=0.7, edgecolors='k')

    # Add labels and title
    plt.xlabel(col_name_x)
    plt.ylabel(col_name_y if col_name_y else "Index")
    plt.title(f"Scatter Plot of {col_name_x} vs {col_name_y or 'Index'}")
    plt.grid(True)
    plt.show()

def box_plot(data, header, col_name):
    """
    Plots a box-and-whisker plot for the specified column in the data.

    Args:
        data (list of lists): The data table as a list of rows.
        header (list of str): The column headers for the data.
        col_name (str): The name of the column to plot.

    Returns:
        None
    """
    # Find column index for the specified column name
    col_index = header.index(col_name)

    # Extract the column values
    col_values = [row[col_index] for row in data]

    # Create a box plot
    plt.figure(figsize=(6, 8))
    plt.boxplot(col_values, vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                medianprops=dict(color="red"))

    # Add labels and title
    plt.ylabel(col_name)
    plt.title(f"Box-and-Whisker Plot of {col_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def calculate_precision_recall_f1(y_test, y_pred, positive_label=1):
    """Calculate the precision, recall, and F1 score.

    Args:
        y_test (list): The true labels.
        y_pred (list): The predicted labels.
        positive_label (int, optional): The label considered as the positive class. Defaults to 1.

    Returns:
        precision (float): The precision of the classifier.
        recall (float): The recall of the classifier.
        f1 (float): The F1 score of the classifier.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Loop through each true and predicted label
    for true, pred in zip(y_test, y_pred):
        if pred == positive_label and true == positive_label:
            true_positive += 1
        elif pred == positive_label and true != positive_label:
            false_positive += 1
        elif pred != positive_label and true == positive_label:
            false_negative += 1

    # Calculate precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def plot_confusion_matrix(y_test, y_pred, labels=None):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        y_test (list): True labels.
        y_pred (list): Predicted labels.
        labels (list, optional): List of unique class labels. If not provided,
                                 it will be inferred from y_test and y_pred.

    Returns:
        None
    """
    # Infer class labels if not provided
    if labels is None:
        labels = sorted(set(y_test) | set(y_pred))

    # Create the confusion matrix
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: i for i, label in enumerate(labels)}

    for true, pred in zip(y_test, y_pred):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index, pred_index] += 1

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.YlGnBu  # Using a different colormap with no white background
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Set axis labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Add values to the cells, changing text color based on background
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            # Choose black or white text color based on cell intensity
            color = "white" if val > matrix.max() / 2 else "black"
            plt.text(j, i, str(val), horizontalalignment="center", color=color)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def calculate_entropy(labels):
    """Calculate entropy for a list of labels."""
    total = len(labels)
    counts = {label: labels.count(label) for label in set(labels)}
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

def calculate_enew(data, attribute_index):
    """Calculate the weighted average of entropies (Enew) for a specific attribute."""
    partitions = {}
    for row in data:
        key = row[attribute_index]
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(row)

    weighted_entropy = sum(
        (len(subset) / len(data)) * calculate_entropy([row[-1] for row in subset])
        for subset in partitions.values()
    )
    return weighted_entropy

def majority_vote(labels):
    """Return the majority label, breaking ties alphabetically."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    majority_label = None
    max_count = 0

    for label, count in sorted(counts.items()):  # Sort labels alphabetically for deterministic result
        if count > max_count or (count == max_count and label < majority_label):
            majority_label = label
            max_count = count

    return majority_label

def tdidt(data, attributes, total_count):
    """Recursive TDIDT algorithm with deterministic leaf result."""
    labels = [row[-1] for row in data]

    # Base case: If all labels are the same, return a leaf
    if len(set(labels)) == 1:
        return ["Leaf", labels[0], len(data), total_count]

    # Base case: If no attributes are left, handle tie-breaking deterministically
    if not attributes:
        # Count occurrences of each label
        label_counts = {label: labels.count(label) for label in set(labels)}
        # Select the label with the highest count, breaking ties alphabetically
        chosen_label = max(label_counts, key=lambda label: (label_counts[label], -ord(label[0])))
        return ["Leaf", chosen_label, len(data), total_count]

    # Use Enew to select the best attribute
    enews = [calculate_enew(data, i) for i in range(len(attributes))]
    best_attribute_index = min(
        range(len(enews)), key=lambda i: (enews[i], attributes[i])
    )  # Break ties by attribute name
    best_attribute = attributes[best_attribute_index]

    # Partition data by the best attribute
    partitions = {}
    for row in data:
        key = row[best_attribute_index]
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(row)

    if len(attributes) == 1:
        tree = ["Attribute",best_attribute]
        for att, value in partitions.items():
            class_label = majority_class(value)
            tree.append(['Value',att,["Leaf",class_label,len(value),len(data)]])
        return tree


    # Build decision node
    tree = ["Attribute", best_attribute]
    for key in sorted(partitions.keys()):  # Sort keys to ensure deterministic order
        subset = partitions[key]
        new_attributes = attributes[:]
        new_attributes.pop(best_attribute_index)
        filtered_subset = [
            [value for col_index, value in enumerate(row) if col_index != best_attribute_index]
            for row in subset
        ]

        # Recursive call to create the subtree
        subtree = tdidt(filtered_subset, new_attributes, len(data))
        tree.append(["Value", key, subtree])

    return tree

def majority_class(instances):
    counts = {}
    for instance in instances:
        label = instance[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return max(counts, key=counts.get)