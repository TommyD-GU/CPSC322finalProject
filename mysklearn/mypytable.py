# pylint: skip-file
"""
###########################################################################
# Programmer: Malia Recker
# Class: CPSC 322 Fall 2024
# Programming Assignment #6
# 11/10/2024
#
# Description: This program defines a class MyPyTable, representing
#   a 2D table of data with column names. The class supports various
#   operations such as loading data from a CSV file, saving data to
#   a file, pretty-printing the table, and performing data transformations
#   like removing rows with missing values, replacing missing values with
#   averages, and computing summary statistics. Additionally, the class
#   implements inner and outer joins between tables based on specified key columns.
#
###########################################################################
"""

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        # Return length of the data list and length of the column_names list
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Find the index of the column based on the identifier
        col_index = self.column_names.index(col_identifier)

        # Initialize an empty list to store the column values
        col = []

        # If missing values are to be included
        if include_missing_values == True:
            # Append all values from the specified column
            for row in self.data:
                col.append(row[col_index])
        else:
            # Append only non-missing values (i.e., values that are not "NA")
            for row in self.data:
                if row[col_index] != "NA":
                    col.append(row[col_index])

        # Return the collected column values
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        # Iterate over each row in the table's data
        for i in range(len(self.data)):
            # Iterate over each element in the row
            for j in range(len(self.data[i])):
                try:
                    # Attempt to convert the element to a float
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    # If conversion fails, skip and continue to the next element
                    continue

        # Return the updated table with numeric values
        return self.data

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # Iterate over the list of row indexes to drop, starting from the largest index
        for index in sorted(row_indexes_to_drop, reverse=True):
            # Remove the row at the specified index from the data
            self.data.pop(index)
    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # Initialize an empty list to store the table data
        table = []

        # Open the CSV file for reading
        infile = open(filename, 'r')
        reader = csv.reader(infile)

        # Read each row from the file and append it to the table list
        for row in reader:
            table.append(row)

        # If the table is non-empty, set column names and the data
        if table:
            self.column_names = table[0]
            self.data = table[1:]

        # Convert all data to numeric where possible
        self.convert_to_numeric()

        # Close the input file
        infile.close()

        # Return the updated table object
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # Open the specified file in write mode with newline handling
        outfile = open(filename, 'w', newline='')
        # Create a CSV writer object for writing to the file
        writer = csv.writer(outfile)

        # Write the header row containing the column names to the CSV
        writer.writerow(self.column_names)

        # Iterate through each data row in the MyPyTable
        for row in self.data:
            # Write the current row to the CSV file
            writer.writerow(row)

        # Close the file after writing all the data
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        # Dictionary to keep track of the rows seen (using keys)
        passed_rows = {}
        # List to store the indices of duplicate rows
        dub_inds = []
        # Find the indexes of the key columns in the data
        key_indexes = [self.column_names.index(col) for col in key_column_names]
        row_index = 0  # Initialize manual row index to keep track of row positions

        # Iterate through each row in the data
        for row in self.data:
            # Create a tuple of key column values for the current row
            key = tuple(row[index] for index in key_indexes)

            # Check if the key has been seen before
            if key in passed_rows:
                # If the key has been seen before, it indicates a duplicate
                dub_inds.append(row_index)
            else:
                # If it's the first time seeing this key, store it with its index
                passed_rows[key] = row_index

            # Manually increment the row index for the next iteration
            row_index += 1

        # Return the list of indices of duplicate rows
        return dub_inds

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        # Initialize an empty list to store rows that do not contain missing values
        new_tbl = []
        # Iterate through each row in the current data
        for row in self.data:
            # Assume the current row is full (i.e., contains no "NA" values)
            row_full = True
            # Check each value in the row to determine if it is missing
            for i in range(len(row)):
                if row[i] == "NA":
                    # If a missing value is found, mark the row as not full
                    row_full = False
            # If the row is full (contains no missing values), add it to the new table
            if row_full:
                new_tbl.append(row)
        # Update the object's data to only include rows without missing values
        self.data = new_tbl
        # Return the updated data
        return self.data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        # Isolate the specified column and retrieve its values
        col = self.get_column(col_name)
        # Get the index of the specified column to facilitate updates later
        col_ind = self.column_names.index(col_name)
        # Initialize variables to calculate the sum of valid numbers and their count
        col_sum = 0
        col_len = 0
        # Iterate through the column values to compute the sum and count of valid numbers
        for val in col:
            try:
                # Attempt to convert the value to a float for numerical operations
                num = float(val)
                # Accumulate the sum of valid numbers
                col_sum += num
                # Increment the count of valid numbers
                col_len += 1
            except (ValueError, TypeError):
                # Skip any values that cannot be converted to float (e.g., "NA" or None)
                continue

        # Calculate the average if there are valid numbers
        if col_len > 0:
            # Average of valid numbers
            col_avg = col_sum / col_len
        else:
            # If no valid numbers, return the data unchanged
            return self.data
        # Replace missing values ("NA") in the original data with the calculated average
        for i in range(len(col)):
            if col[i] == "NA":
                # Update the corresponding entry in the data
                self.data[i][col_ind] = col_avg
        # Return the updated data with missing values replaced
        return self.data


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        # Convert column values to numeric types if applicable
        self.convert_to_numeric()
        # Create a summary table with specified headers for the statistics
        summary = MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], [])
        # Iterate through each specified column to compute statistics
        for attrib in col_names:
            col_sum = [attrib]  # Start the summary with the column name
            # Get the column values, excluding missing values
            col_unsorted = self.get_column(attrib, include_missing_values=False)
            # Sort the column values for statistical calculations
            col = sorted(col_unsorted)  # If the column is empty, nothing will be computed
            # Proceed only if the column has data
            if len(col) > 0:
                # Calculate the minimum and maximum values
                col_min = min(col)
                col_max = max(col)

                # Calculate the mid-point between min and max
                col_mid = (col_min + col_max) / 2
                # Calculate the average of the column
                col_avg = sum(col) / len(col)
                # Calculate the median
                n = len(col)
                # Odd number of elements
                if n % 2 == 1:
                    median_val = col[n // 2]  # Middle element
                else:  # Even number of elements
                    # Average of the two middle elements
                    median_val = (col[(n // 2) - 1] + col[n // 2]) / 2.0
                # Append all calculated statistics to the summary list
                col_sum.extend([col_min, col_max, col_mid, col_avg, median_val])
                # Append the summary of the current column to the summary table
                summary.data.append(col_sum)
        return summary  # Return the summary table with statistics
    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Create a new MyPyTable instance to store the result of the join
        joined_tbl = MyPyTable()

        # Retrieve the header and data from the other table
        other_header = other_table.column_names
        other_data = other_table.data

        # Get the indexes of the key columns in both tables
        key_ind = [self.column_names.index(col) for col in key_column_names]
        other_key_ind = [other_header.index(col) for col in key_column_names]

        # Set the column names for the joined table by combining both table headers,
        # excluding the key columns from the other table
        rest_of_attribs = [other_header[i] for i in range(len(other_header)) if i not in other_key_ind]
        joined_tbl.column_names = self.column_names + rest_of_attribs

        # Prepare a list to store the joined rows
        joined_rows = []

        # Perform the inner join by iterating through each row in self.data
        for row in self.data:
            # Extract the key for the current row in the first table
            key_self = tuple(row[index] for index in key_ind)

            # Compare it with rows from the other table
            for other_row in other_data:
                # Extract the key for the current row in the other table
                key_other = tuple(other_row[index] for index in other_key_ind)

                # If the keys match, combine the rows, excluding the key columns from the other table
                if key_self == key_other:
                    combined_row = row + [other_row[i] for i in range(len(other_row)) if i not in other_key_ind]
                    joined_rows.append(combined_row)

        # Assign the joined rows to the new table
        joined_tbl.data = joined_rows

        # Return the resulting joined table
        return joined_tbl

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        # Create a new MyPyTable instance to store the result of the join
        joined_tbl = MyPyTable()

        # Retrieve the header and data from the other table
        other_header = other_table.column_names
        other_data = other_table.data

        # Get the indexes of the key columns in both tables
        key_ind = [self.column_names.index(col) for col in key_column_names]
        other_key_ind = [other_header.index(col) for col in key_column_names]

        # Set the column names for the joined table by combining both table headers,
        # excluding the key columns from the other table
        rest_of_attribs = [other_header[i] for i in range(len(other_header)) if i not in other_key_ind]
        joined_tbl.column_names = self.column_names + rest_of_attribs

        # Prepare a list to store the joined rows
        joined_rows = []

        # Create sets to track which rows from both tables have been joined
        joined_self_rows = set()
        joined_other_rows = set()

        # Perform the outer join by first matching rows with equal keys
        for i, row in enumerate(self.data):
            # Extract key for the current row in the first table
            key_self = tuple(row[index] for index in key_ind)

            # Compare it with rows from the other table
            for j, other_row in enumerate(other_data):
                # Extract key for the current row in the other table
                key_other = tuple(other_row[index] for index in other_key_ind)

                # If the keys match, combine the rows
                if key_self == key_other:
                    combined_row = row + [other_row[i] for i in range(len(other_row)) if i not in other_key_ind]
                    joined_rows.append(combined_row)

                    # Mark rows as joined
                    joined_self_rows.add(i)
                    joined_other_rows.add(j)

        # Add rows from self.data that didn't have a match in other_table
        for i, row in enumerate(self.data):
            if i not in joined_self_rows:
                # Pad the row with "NA" for columns from other_table
                padded_row = row + ["NA"] * len(rest_of_attribs)
                joined_rows.append(padded_row)

        # Add rows from other_table.data that didn't have a match in self.data
        for j, other_row in enumerate(other_data):
            if j not in joined_other_rows:
                # Create a row padded with "NA" for the columns from self.table
                padded_row = ["NA"] * len(self.column_names)

                # Insert the key columns from other_table into the correct positions
                for idx, key_index in enumerate(other_key_ind):
                    padded_row[key_ind[idx]] = other_row[key_index]

                # Add the non-key columns from other_table
                padded_row += [other_row[k] for k in range(len(other_row)) if k not in other_key_ind]
                joined_rows.append(padded_row)

        # Assign the joined rows to the new table
        joined_tbl.data = joined_rows

        # Return the resulting joined table
        return joined_tbl
