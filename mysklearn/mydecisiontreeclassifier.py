import utils
import numpy as np

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
        self.header = None
        self.attribute_domains = None

    def calculate_entropy(self, partition):
        """
        Calculate entropy for a given partition.

        Entropy is a measure of the amount of uncertainty or impurity in a dataset.
        This function calculates the entropy of a partition by summing the
        proportion of each label in the partition multiplied by the log base 2
        of that proportion, and then taking the negative of that sum.

        Parameters:
        partition (list of lists): A list of instances, where each instance is a
                                   list and the last element is the label.

        Returns:
        float: The entropy of the partition. If the partition is empty, returns 0.
        """
        total = len(partition)
        if total == 0:
            return 0
        counts = {}
        for instance in partition:
            label = instance[-1]
            counts[label] = counts.get(label, 0) + 1
        return -sum((count / total) * np.log2(count / total) for count in counts.values())


    def select_attribute(self, instances, attributes):
        """
        Selects the best attribute to split the instances based on the minimum entropy.
        Args:
            instances (list of list): The dataset where each inner list represents an instance.
            attributes (list of str): The list of attribute names to consider for splitting.
        Returns:
            str: The attribute with the lowest weighted entropy, indicating the best split.
        """
        min_entropy = float('inf')
        best_attribute = None

        for attribute in attributes:
            partitions = self.partition_instances(instances, attribute)
            weighted_entropy = 0
            total_instances = len(instances)

            for partition in partitions.values():
                weighted_entropy += (len(partition) / total_instances) * self.calculate_entropy(partition)

            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = attribute

        return best_attribute

    def partition_instances(self, instances, attribute):
        """
        Partitions a list of instances based on the values of a specified attribute.
        Args:
            instances (list of list): The dataset to be partitioned, where each inner list represents an instance.
            attribute (str): The attribute on which to partition the instances.
        Returns:
            dict: A dictionary where keys are attribute values and values are lists of instances that have the corresponding attribute value.
        """

        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def all_same_class(self, instances):
        """
        Check if all instances belong to the same class.

        Args:
            instances (list of list): A list of instances, where each instance is a list and the last element is the class label.

        Returns:
            bool: True if all instances have the same class label, False otherwise.
        """
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True

    def tdidt(self, current_instances, available_attributes):
        """
        Perform the Top-Down Induction of Decision Trees (TDIDT) algorithm to build a decision tree.
        Args:
            current_instances (list of list): The current subset of instances to be used for building the tree.
            available_attributes (list): The list of attributes that can be used for splitting.
        Returns:
            list: A nested list representing the decision tree. The tree is built recursively and contains nodes
                  of the form ["Attribute", attribute_name], ["Value", attribute_value], and ["Leaf", class_label,
                  class_count, total_count].
        The function follows these steps:
            1. Select an attribute to split on.
            2. Remove the selected attribute from the list of available attributes.
            3. Create a tree node for the selected attribute.
            4. Partition the instances based on the selected attribute's values.
            5. For each partition, check for base cases:
                - All class labels in the partition are the same: create a leaf node.
                - No more attributes to select: create a majority vote leaf node.
                - No more instances to partition: create a majority vote leaf node.
            6. If none of the base cases are met, recursively build the subtree for the partition.
            7. Append the subtree to the current tree node.
        """
        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same
            # => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                class_label = att_partition[0][-1]
                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            #    CASE 2: no more attributes to select (clash)
            # => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                class_labels = [row[-1] for row in current_instances]
                label_counts = {label: class_labels.count(label) for label in set(class_labels)}

                # Sort labels alphabetically, then by count (highest count first)
                sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
                class_label = sorted_labels[0][0]

                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            #    CASE 3: no more instances to partition (empty partition)
            # => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                class_labels = [row[-1] for row in current_instances]
                label_counts = {label: class_labels.count(label) for label in set(class_labels)}

                # Sort labels alphabetically, then by count (highest count first)
                sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
                class_label = sorted_labels[0][0]
                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            else:
                # none of base cases were true, recurse!!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree



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
        self.header = [f"att{i}" for i in range(len(X_train[0]))]
        self.attribute_domains = {
            self.header[i]: list(set(row[i] for row in X_train))
            for i in range(len(X_train[0]))
        }

        combined_data = [x + [y] for x, y in zip(X_train, y_train)]
        self.tree = self.tdidt(combined_data, self.header[:])

    def tdidt_predict(self, tree, instance, header):
        """
        Predict the class label for a given instance using a decision tree.
        Parameters:
        tree (list): The decision tree represented as a nested list.
        instance (list): The instance to classify.
        header (list): The list of attribute names corresponding to the instance.
        Returns:
        The predicted class label for the given instance.
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1] # class label

        # if we are here, we are at an Attribute
        # we need to match the instance's value for this attribute
        # to the appropriate subtree
        att_index = header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance, header)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []

        for instance in X_test:
            y_pred.append(self.tdidt_predict(self.tree, instance, self.header))
        return y_pred

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
        if attribute_names is None:
            attribute_names = self.header

        def traverse(tree, conditions):
            if tree[0] == "Leaf":
                # Base case: We've reached a leaf
                label = tree[1]
                count = tree[2]
                total = tree[3]
                rule = " AND ".join(conditions)
                print(f"IF {rule} THEN {class_name} = {label} [{count}/{total}]")
            elif tree[0] == "Attribute":
                # Recursive case: Traverse each value subtree
                attribute = attribute_names[self.header.index(tree[1])]
                for value_subtree in tree[2:]:
                    value = value_subtree[1]
                    traverse(value_subtree[2], conditions + [f"{attribute} == {value}"])

        traverse(self.tree, [])
