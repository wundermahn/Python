# csv library used to import data
import csv
# math library used for some mathematical and statistical functions
import math
# random library used for shuffling and randomizing data
import random
# numpy library used for some data management / some statistical processes
import numpy
# copy used for deepcopy, which is needed for strings since the string class doesnt contain a copy ability
import copy
# sys library used to accept command line arguments to the program
import sys

#################### CLASSES LIBRARY ####################

# Creating an OO-type class for a decision tree
class DecisionTree():
    # Create an empty tree
    tree = None

    # A decision tree "learns" by building it
    # It needs a set of data, attributes (features), and a target variable (the class)
    def learn(self, training_set, attributes, target):
        self.tree = grow_tree(training_set, attributes, target)


# Creating an OO-type class for a node of a tree
class Node():
    # Create the variables or objects of a node
    # The evaluated feature
    attr = ""
    # The relevant values
    values = ""
    # The child nodes (subtrees, normally)
    children = None
    # The dominant class
    biggest = ""

    # Initiate a node using the following parameters
    def __init__(self, attr, val, tree, biggest):
        self.attr = attr
        self.values = val
        self.children = tree
        self.biggest = biggest

#################### CLASSES LIBRARY ####################

################# DATA HANDLING LIBRARY #################

# This function attempts to turn a csv into an array
def csv_to_array(filepath):
    # Create a blank array in which we will hold the data
    data = []
    # Create a blank array in which we will hold the attributes
    attributes = []
    # Create a boolean to check to see if you are on the first row
    # Set it to True at first
    first_row = True
    # Open the file and begin to loop through it
    with open(filepath) as file:
        # Loop through the file
        for row in csv.reader(file, delimiter=","):
            # If you are on the first line
            if first_row == True:
                # Set the attributes variable
                attributes = row
                # Now flip it to false
                first_row = False
            # Since the counter is now above 0...
            else:
                # add the rows of the csv to the data array
                data.append(tuple(row))
    # Return the attributes (features) and the data
    return (attributes, data)


# This function prepares the data by extracting the attributes (column headings) and the target, which is the classes column (or whatever classification column there is)
def prepare_data(file):
    # Create an array of the now read in data
    data = csv_to_array(file)
    # Create a list of the attributes, which are the column headings
    attributes = data[0]
    # Create the target, which is the last column in the array, which is the classification column
    target = attributes[-1]
    # "Good data" is the actual data itself
    good_data = data[1]
    # Shuffle that in place
    random.shuffle(good_data)

    # Return these things
    return good_data, attributes, target


# Function to split the data into test, training set, and validation sets
def split_data(csv):
    # Grab the number of rows and calculate where to split
    temp_csv = numpy.asarray(csv)
    num_rows = temp_csv.shape[0]
    validation_split = int(num_rows * 0.10)
    training_split = int(num_rows * 0.72)

    # Validation set as the first 10% of the data
    validation_set = csv[:validation_split]
    # Training set as the next 72
    training_set = csv[validation_split:training_split + validation_split]
    # Testing set as the last 18
    testing_set = csv[training_split + validation_split:]

    # Return the datasets
    return testing_set, training_set, validation_set

################# DATA HANDLING LIBRARY #################

# This function returns the list of classes, and their associated weights (i.e. distributions)
# for a given dataset
def class_distribution(dataset):
    # Ensure the dataset is a numpy array
    dataset = numpy.asarray(dataset)
    # Collect # of total rows and columns, using numpy
    num_total_rows = dataset.shape[0]
    num_columns = dataset.shape[1]
    # Create a numpy array of just the classes
    classes = dataset[:, num_columns - 1]
    # Use numpy.unique to remove duplicates
    classes = numpy.unique(classes)
    # Create an empty array for the class weights
    class_weights = []

    # Loop through the classes one by one
    for aclass in classes:
        # Create storage variables
        total = 0
        weight = 0
        # Now loop through the dataset
        for row in dataset:
            # If the class of the dataset is equal to the current class you are evaluating, increase the total
            if numpy.array_equal(aclass, row[-1]):
                total = total + 1
            # If not, continue
            else:
                continue
        # Divide the # of occurrences by total rows
        weight = float((total / num_total_rows))
        # Add that weight to the list of class weights
        class_weights.append(weight)

    # Turn the weights into a numpy array
    class_weights = numpy.asarray(class_weights)
    # Return the array
    return classes, class_weights

# This functions performs k cross fold validation for classification
def cross_fold_validation_classification(dataset, k):
    # Create a numpy copy of the dataset
    temp_dataset = numpy.asarray(dataset)
    # Grab the classes and class weights (their occurence distributions) across thed ataset
    classes, class_weights = class_distribution(temp_dataset)
    # Grab the total length of the dataset
    total_num_rows = temp_dataset.shape[0]
    # Create a copy of the dataset to actually manipulate
    data = numpy.copy(temp_dataset)
    # Create an array with hold the k folds
    total_fold_array = []

    # Basic for loop to iterate k times
    for _ in range(k):
        # Set an array of the current fold being created
        curr_fold_array = []

        # Loop through each class and its associated weight
        for a_class, a_class_weight in zip(classes, class_weights):
            # Shuffle the remaining data
            numpy.random.shuffle(data)
            # Keep track of how many items have been added
            num_added = 0
            # Determine hom many items to add, based on the distribution in the respective classes
            num_to_add = float((((a_class_weight * total_num_rows)) / k))
            # Keep an index to know which rows to potentially delete
            tot = 0
            # Loop through the data array
            for row in data:
                # Set the current value to be the object in the last column, which is the classification
                curr = row[-1]
                # If you have already added what you were supposed to, stop. If not...
                if num_added >= num_to_add:
                    break
                else:
                    # If the class is the same as that for the current row
                    if (a_class == curr):
                        # Add that row to the current fold
                        curr_fold_array.append(row)
                        # Increase the number of items you've added
                        num_added = num_added + 1
                        # Delete that row from the dataset
                        numpy.delete(data, tot)
                # Increase your index
                tot = tot + 1
        # Append the fold to the array
        total_fold_array.append(curr_fold_array)

    # Return the folds
    return total_fold_array

# This function checks to see if a feature is continuous
def isContinuous(feature):
    # Loop through the values in the feature (which is passed in as an array)
    for val in feature:
        # If its a string return nothing, that means its a class or something else that is discrete
        if (val is type(val) == str):
            return null
        # If any value in the feature has trailing numbers (i.e., it is continuous)
        if (feature - math.floor(val) > 0):
            # Return true
            return true;
        # Otherwise, keep going
        else:
            continue
    # If you got here than that means that none of the values returned true, so therefore they must be false
    return false

# Function to return Shannon's Entropy
def entropy(attributes, dataset, targetAttr):
    # Set a blank array of frequencies (needed to calculate probability)
    freq = {}
    # Set an initial entropy value to 0
    entropy = 0.0
    # Set an index to 0
    index = 0
    # Loop through each attribute
    for item in attributes:
        # If the current attribute is the one we care about, break
        if (targetAttr == item):
            break
        # Otherwise, increase the index
        else:
            index = index + 1

    # reduce the index
    index = index - 1

    # Now, loop through the data
    for item in dataset:
        # If that item is in frequencies, update its frequency (i.e. number of occurrences)
        if ((item[index]) in freq):
            # Increase the index
            freq[item[index]] += 1.0
        # Else...
        else:
            # Initialize it by setting it to 0
            freq[item[index]] = 1.0

    # Now loop through the frequency array values
    for freq in freq.values():
        # Calculate the entropy, making sure we use log base 2
        entropy = entropy + (-freq / len(dataset)) * math.log(freq / len(dataset), 2)
    # Return the entropy
    return entropy

# Calculates the information gain ratio
def IG(attributes, dataset, attr, targetAttr):
    # Set a blank array of frequencies (needed to calculate probability)
    freq = {}
    # Initialize an entropy value for the subset of the data we are testing
    sub_entropy = 0.0
    # Grab the index of the attribute that we care about
    index = attributes.index(attr)
    # Grab the total entropy of the dataset
    total_entropy = entropy(attributes, dataset, targetAttr)

    # Loop through the data
    for item in dataset:
        # If that is in the frequency array, increase its frequency
        if ((item[index]) in freq):
            freq[item[index]] += 1.0
        # If it is not, initialize its frequency
        else:
            freq[item[index]] = 1.0

    # Loop through the frequency values
    for currFreq in freq.keys():
        # Grab the probablity of that specific currFrequency
        valProb = freq[currFreq] / sum(freq.values())
        # Grab the subset of the data we care about
        dataSubset = [entry for entry in dataset if entry[index] == currFreq]
        # Calculate the entropy of the subset
        sub_entropy = sub_entropy + (valProb * entropy(attributes, dataSubset, targetAttr))

    # Return the information gain ratio
    return (total_entropy - sub_entropy)

# This function returns unique values for a given attribute in the dataset
def get_unique_values(dataset, attributes, attr):
    # Grab the index of the attribute you care about
    index = attributes.index(attr)
    # Create an array of the values
    vals = []

    # Loop through the dataset
    for item in dataset:
        # If the current item (for the given attribute) isn't currently in values, add it
        if item[index] not in vals:
            vals.append(item[index])
    # Return values
    return vals

# This function returns the best attribute in the given dataset as defined by information gain ratio
def choose_best_attribute(dataset, attributes, target):
    # Set the current best to an arbitrary value (just the first)
    best_attribute = attributes[0]
    # Create a variable to hold the best gain, and initialize it to 0
    best_gain = 0

    # Loop through all of the attributes
    for currAttr in attributes:
        # Calculate the current gain for each attribute
        currGain = IG(attributes, dataset, currAttr, target)
        # If the current gain is better than the best gain
        if currGain > best_gain:
            # Set the best gain to the current
            best_gain = currGain
            # Set the best attribute to the current attribute being evaluated
            best_attribute = currAttr

    # Return the best attribute
    return best_attribute

# This function collects all of the data for the given best_attribute
def get_bestattr_data(dataset, attributes, best_attr, val):
    # Create a new 2d array to be filled
    good_data = [[]]
    # Grab the index of the best attribute
    index = attributes.index(best_attr)

    # For each item in the dataset
    for item in dataset:
        # if that curr value is equal to the value we care about
        if (item[index] == val):
            # Create a new "array"
            temp_array = []
            # Loop through the current row
            for jindex in range(0, len(item)):
                # If the current count that you are on is not the index
                if (jindex != index):
                    # Append it
                    temp_array.append(item[jindex])
            # Now append that temp array to good_data
            good_data.append(temp_array)
    # Remove any blank or empty arrays from the 2d array
    good_data.remove([])
    # Return the 2d array
    return good_data

# This function returns the class with the most rows associated with it
def biggest_class(attributes, dataset, target):
    # Create an array of frequencies (i.e. number of occurrences)
    freq = {}
    # Grab the index of the target attribute
    index = attributes.index(target)

    # Loop through the dataset in a tuple format
    for tuple in dataset:
        # If that tuple for the index is already in the frequency array, increase its frequency
        if ((tuple[index]) in freq):
            freq[tuple[index]] += 1
        # Else, set its frequency to 1 and initialize it
        else:
            freq[tuple[index]] = 1

    # Set blank variables (initializing them to nothing) for the biggest frequency and the biggest class
    biggest_frequency = 0
    biggest_class = ""

    # Loop through the "keys" in the frequency array, the values
    for key in freq.keys():
        # If that frequency is bigger than the biggest so far
        if freq[key] > biggest_frequency:
            # Set the biggest frequency to the current
            biggest_frequency = freq[key]
            # Set the biggest class to the current
            biggest_class = key

    # Return the biggest class
    return biggest_class

# This function is used to build the decision tree using the given data, attributes and the target attributes. It returns the decision tree in the end.
def grow_tree(dataset, attributes, target):
    # Create a copy of the dataset that was passed in so we dont edit the good dataset
    data = dataset[:]
    # Create a list of values
    values = [record[attributes.index(target)] for record in data]
    # Create the "biggest", which is the biggest and most dominating class, which will end up being the root
    biggest = biggest_class(attributes, data, target)

    # If you've run out of data
    if ((not data) or ((len(attributes) - 1)) <= 0):
        # Return a blank tree
        tree = Node("class", biggest, None, biggest)
        return tree

    # If you have a completely pure node, meaning its all the same class
    elif values.count(values[0]) == len(values):
        # Create a pure node
        tree = Node("class", values[0], None, biggest)
        # And return it
        return tree

    # If neither of those are true, then you are ready to build a real tree
    else:
        # Determine your best attribute
        best_attribute = choose_best_attribute(data, attributes, target)
        # Create a blank tree
        tree = []
        # Create an array of values
        values = []

        # Loop through the current values
        for val in get_unique_values(data, attributes, best_attribute):
            # Create an array of good data
            good_data = get_bestattr_data(data, attributes, best_attribute, val)
            # Create a copy of the attributes
            new_attribute = attributes[:]
            # Remove the best attribute from the current copy
            new_attribute.remove(best_attribute)
            # Now build a subtree for this node
            subtree = grow_tree(good_data, new_attribute, target)
            # Append on the subtree, essentially making a child
            tree.append(subtree)
            # Append the current value
            values.append(val)

    # Return the head of your tree, which will then contain the rest of the tree as child objects (recursively)
    return Node(best_attribute, values, tree, biggest)

# This function will recurse down the tree and once it reaches a leaf node determine
# if it got the correct answer
def recurse_tree(tree, test_entry, attributes):
    # If you have no children you are a leaf
    if not tree.children:
        # If the class of the leaf matches the test
        if test_entry[-1] == tree.biggest:
            # Return a 1 which is true
            return 1
        # Other wise you are incorrect
        else:
            # Return a 0 which is incorrect
            return 0
    # Get the index of the attribute you are about to recurse on
    index = attributes.index(tree.attr)
    # Get the value of the attribute of the test entry
    value = test_entry[index]
    # Set a counter
    count = 0
    # For each value in the tress values
    for val in tree.values:
        # If you found the corresponding value to the test entry
        if value == val:
            # Recurse down
            return recurse_tree(tree.children[count], test_entry, attributes)
        # Increment counter
        count += 1

# This function prints out the current decision tree
def print_tree(tree):
    # If there are no children
    if (not tree.children):
        # Then you just have a root node. Print that out and alert the enduser
        print("No children, root node")
    # If you do have children (therefore a tree...)
    else:
        # What level you are at
        counter = 0
        # loop through each "child"
        for child in tree.children:
            # Print out their value
            print("Child Value: ", counter, child.values)
            # Increase the counter
            counter = counter + 1

# This function detects if the next subtree has children
def next_has_children(subtree):
    # If it does, return true
    if (subtree.children == True):
        return True
    # If it does not, return false
    else:
        return False

# This function removes the children of a subtree
def remove_children(subtree):
    # Return the subtree with all children removed
    return subtree.children == None

# This function actually prunes the tree that is passed in
def prune_tree(tree):
    # Set a temp tree to be a copy of the one passedin
    temp_tree = tree
    # If it does not have any children, i.e. its the root
    if (not temp_tree.children):
        # Return it
        return temp_tree
    # If it does have children:
    else:
        # Go through the children
        for child in temp_tree.children:
            # If the child has children (i.e. the tree is even longer)
            if (next_has_children(child)):
                # Determine the "next"
                next = child.children
                # Prune the next
                prune_tree(next)
            else:
                # Otherwise, if it does not, remove the current child's children
                remove_children(child)

    # Return the pruned tree
    return temp_tree

# This function regrows a pruned tree
def grow_pruned_tree(pruned_tree, dataset, attributes, target):
    # Create a copy of the dataset that was passed in so we dont edit the good dataset
    data = dataset[:]
    # Randomly shuffle the dataset
    random.shuffle(data)
    # Create a list of values
    values = [record[attributes.index(target)] for record in data]
    # Create the "biggest", which is the biggest and most dominating class, which will end up being the root
    biggest = biggest_class(attributes, data, target)

    # If you've run out of data
    if ((not data) or ((len(attributes) - 1)) == 0):
        # Return a blank tree
        pruned_tree = Node("class", biggest, None, biggest)
        return pruned_tree

    # If you have a completely pure node, meaning its all the same class
    elif values.count(values[0]) == len(values):
        # Create a pure node
        pruned_tree = Node("class", values[0], None, biggest)
        # And return it
        return pruned_tree

    # If neither of those are true, then you are ready to build a real tree
    else:
        # Determine your best attribute
        best_attribute = choose_best_attribute(data, attributes, target)
        # Create a blank array to hold the newly pruned tree
        pruned_tree_arr = []
        # Create an array of values for the newly pruned tree
        pruned_values = []

        # Loop through the current values
        for val in get_unique_values(data, attributes, best_attribute):
            # Create an array of good data
            good_data = get_bestattr_data(data, attributes, best_attribute, val)
            # Create a copy of the attributes
            new_attribute = attributes[:]
            # Remove the best attribute from the current copy
            new_attribute.remove(best_attribute)
            # Now build a subtree for this node
            subtree = grow_pruned_tree(pruned_tree, good_data, new_attribute, target)
            # Append on the subtree, essentially making a child, to the newly pruned tree
            pruned_tree_arr.append(subtree)
            # Append the current pruned value
            pruned_values.append(val)

    # Return the head of your tree, which will then contain the rest of the tree as child objects (recursively)
    return Node(best_attribute, pruned_values, pruned_tree_arr, biggest)

# This function actually creates, runs, and tests the newly created decision tree
def run_decision_tree(dataset, attributes, target):
    # Randomize the data
    random.shuffle(dataset)
    # Create testing, training, and validation sets of data
    testing_set, training_set, validation_set = split_data(dataset)
    # Create a blank decision tree
    tree = DecisionTree()
    # Grow the decision tree with the training data
    tree.learn(training_set, attributes, target)
    # Create an array to hold the results of the testing
    results = []

    # Loop through the testing set
    for entry in testing_set:
        # Recurse down the tree looking for the given testing entry
        answer = recurse_tree(tree.tree, entry, attributes)
        # If it classified, append it
        if answer != None:
            results.append(answer)
        # If it did not, just continue through
        else:
            continue
    # Set the accuracy to be the count of "1s" in the results array (correctly classified) over the total # of items
    accuracy = float(results.count(1)) / float(len(results))
    # Print out the accuracy
    print("UNPRUNED ACCURACY: ", accuracy * 100)

    # Create an array to hold the results of the testing for the pruned tree
    pruned_results = []
    # Create a pruned decision tree, based off of the one that was testedabove
    pruned_tree = prune_tree(tree.tree)
    # That was just a semi-blank tree, now grow it out as needed
    new_pruned_tree = grow_pruned_tree(pruned_tree, dataset, attributes, target)
    # Loop through the validation set
    for entry in validation_set:
        # Recurse down the tree looking for the given testing entry
        pruned_answer = recurse_tree(new_pruned_tree, entry, attributes)
        # If it classified, append it
        if pruned_answer != None:
            pruned_results.append(pruned_answer)
        # If it did not, just continue through
        else:
            continue
    # Set the accuracy to be the count of "1s" in the results array (correctly classified) over the total # of items
    pruned_accuracy = float(pruned_results.count(1)) / float(len(pruned_results))
    # Print out the accuracy
    print("PRUNED ACCURACY: ", pruned_accuracy * 100)

    # Return the accuracies
    return accuracy, pruned_accuracy

# This function builds a decision tree in a k cross fold validating stat technique
def run_decision_tree_cfv(dataset, attributes, target, k):
    # Initiate a variable to capture the best accuracy, and a total to average all accuracies
    best_accuracy = 0
    accuracy_total = 0
    # Initiate a variable to capture the best accuracy for the pruned trees, and a total to average all pruned accuracies
    best_pruned_accuracy = 0
    pruned_accuracy_total = 0

    # Create the folds from the given dataset
    folds = cross_fold_validation_classification(dataset, k)
    # Initiate running count of folds
    fold_num = 1

    # Loop through the folds
    for fold in folds:
        # Randomize the data
        random.shuffle(dataset)
        # Spacing for end user / terminal output
        print("                   ")
        # Update user as to what fold is currently being tested
        print("CURRENT FOLD: ", fold_num)
        # Create training, testing, and validation sets for the fold
        testing_set, training_set, validation_set = split_data(fold)
        # Grab the current fold's accuracy
        accuracy, pruned_accuracy = run_decision_tree(training_set, attributes, target)
        # Add to the totals
        accuracy_total = accuracy_total + accuracy
        pruned_accuracy_total = pruned_accuracy_total + pruned_accuracy
        # Increment the fold counter
        fold_num = fold_num + 1
        # If the accuracy for this run was better than the best so far
        if accuracy > best_accuracy:
            # Update the best to reflect this fold's accuracy
            best_accuracy = accuracy
        # If it was not, just continue
        else:
            continue
        # Now check your pruned accuracies
        # If the pruned accuracy for this run was better than the best so far
        if pruned_accuracy > best_pruned_accuracy:
            # Update the best to reflect this fold's pruned accuracy
            best_pruned_accuracy = pruned_accuracy
        # If it was not, just continue
        else:
            continue

    print("                 ")

    # Unpruned Data
    print("YOUR BEST ACCURACY WAS: ", best_accuracy * 100)
    print("YOUR AVERAGE ACCURACY WAS: ", ((accuracy_total / k) * 100))

    # Pruned Data
    print("YOUR BEST PRUNED ACCURACY WAS: ", best_pruned_accuracy * 100)
    print("YOUR AVERAGE PRUNED ACCURACY WAS: ", ((pruned_accuracy_total / k) * 100))

# Main function
def main():
    # Collect the abalone data from the command line argument
    abalone = sys.argv[1]
    # Collect the car data from the command line argument
    car = sys.argv[2]
    # Collect the image data from the command line argument
    image = sys.argv[3]

    # Test Abalone
    print("NOW TESTING: ABALONE")
    # Prepare the data that was passed in as a csv file
    abalone_data, abalone_attributes, abalone_target = prepare_data(abalone)
    # Create, run, and test the decision tree using 5-fold cfv
    run_decision_tree_cfv(abalone_data, abalone_attributes, abalone_target, 5)

    print("*************************************")

    # Test Car
    print("NOW TESTING: CAR")
    # Prepare the data that was passed in as a csv file
    car_data, car_attributes, car_target = prepare_data(car)
    # Create, run, and test the decision tree using 5-fold cfv
    run_decision_tree_cfv(car_data, car_attributes, car_target, 5)

    print("*************************************")

    # Test Image Segmentation
    print("NOW TESTING: IMAGE SEGMENTATION")
    # Prepare the data that was passed in as a csv file
    image_data, image_attributes, image_target = prepare_data(image)
    # Create, run, and test the decision tree using 5-fold cfv
    run_decision_tree_cfv(image_data, image_attributes, image_target, 5)

# Call the main function
main()
