from random import random
# randrange used for randomly selecting elements to add to fold
from random import randrange
# csv used to read and import data
from csv import reader
# math used for functions like exponentials
from math import exp
# numpy used for linear algebra functions and some data handling
import numpy
# sys used to accept command line arguments to the program
import sys
# statistics and mean used to calculate averages
from statistics import mean


################# DATA HANDLING LIBRARY #################

# This functions handles the raw data and returns a dataset ready to be used
def handle_data(file):
    # Load the csv file into an array
    data = csv_to_array(file)
    # Format the csv file (changing types)
    dataset = format_data(data)
    # Normalize the data as needed
    normalize_data(dataset)

    # Return the type formatted and randomized dataset
    return dataset


# This function normalizes the data for use in the backprop algorithm
def normalize_data(dataset):
    # Determine the minimum and maximum and store them in an array
    minimum_maximum = [[min(column), max(column)] for column in zip(*dataset)]
    # Loop through the dataset
    for row in dataset:
        # Normalize every column except for the last (classifications) column
        for i in range(len(row) - 1):
            # Perform normalization
            row[i] = (row[i] - minimum_maximum[i][0]) / (minimum_maximum[i][1] - minimum_maximum[i][0])


# This function formats the dataset into the correct type values
def format_data(dataset):
    # Create a temporary version of the dataset passed in
    temp = dataset
    # Determine the number of columns
    num_cols = len(temp[0]) - 1
    # Loop through the columns
    for col in range(num_cols):
        # Convert them to floats
        convert_to_float(temp, col)

    # Convert the classifications column to ints
    convert_to_int(temp, num_cols)

    # Return the array
    return temp

# This function attempts to turn a csv into an array
def csv_to_array(filename):
    # Create a blank array in which we will hold the data
    dataset = []
    # Open the file
    with open(filename, 'r') as file:
        # Use csv library to read the file
        csv_reader = reader(file)
        # Loop through the reader and append the row to the array we created earlier
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    # Return the dataset
    return dataset


# This function converts objects in a column to floats
def convert_to_float(dataset, column):
    # Loop through the dataset
    for row in dataset:
        # Convert all items in that column to floats
        row[column] = float(row[column])

    # NOTE - Used to avoid needing to convert everything and then reconvert the classification column

# This function converts objects in a column to ints
def convert_to_int(dataset, column):
    # Create an array of the class values
    class_values = [row[column] for row in dataset]
    # Determine a set of unique classes
    unique_classes = set(class_values)
    # Create an index (a dict structure)
    index = dict()
    # Loop through the unique classes
    for iterator, item in enumerate(unique_classes):
        # Set the index of the class to be equal to the index in the dictionary
        index[item] = iterator
    # Now loop through the dataset
    for row in dataset:
        # Set the column of that row to be the value in the lookup dictionary
        row[column] = index[row[column]]

    # Return the dictionary we created
    return index

# Split a dataset into k folds
def create_k_folds(dataset, k):
    # Create the empty folds array
    folds = []
    # Createa copy of the dataset
    copy = dataset
    # Determine the number of elements to use for each fold
    num_ele = int(len(copy) / k)

    # Loop through k times
    for index in range(k):
        # Create a temp array for the current fold
        currFold = []
        # While the current fold does not have the proper amount of elements
        while len(currFold) < num_ele:
            # Create a random index using the randrange from the random library
            item = randrange(len(copy))
            # Append that item to the current fold by popping it out of the dataset
            currFold.append(copy.pop(item))
        # Append the current fold to the folds array
        folds.append(currFold)

    # Return the folds array
    return folds

# Calculate accuracy percentage
def get_backprop_accuracy(testing_classes, predictions):
    # Set a variable to collect the # of correct guesses
    num_correct = 0
    # Loop through each of the actual values
    for index in range(len(testing_classes)):
        # If the actual value is the same as your prediction / guess
        if (testing_classes[index] == predictions[index]):
            # Increase the number of correct
            num_correct = num_correct + 1
        # Otherwise
        else:
            # Continue on
            continue

    # Return a float of the num correct / total values
    return float((num_correct / len(testing_classes)) * 100)

# Evaluate an algorithm using a cross validation split
def run_backprop(dataset, k, learning_rate, epochs, hidden_nodes):
    folds = create_k_folds(dataset, k)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = back_propagation(train_set, test_set, learning_rate, epochs, hidden_nodes)
        actual = [row[-1] for row in fold]
        accuracy = get_backprop_accuracy(actual, predicted)
        scores.append(accuracy)

    return float(mean(scores))

def tune_backprop(dataset, k):
    learning_rate = 0
    epochs = 0
    hidden_nodes = 0

    best_accuracy = 0
    best_learning_rate = 0
    best_epoch = 0
    best_hidden_nodes = 0

    for _ in range(6):
        learning_rate = learning_rate + .05
        epochs = epochs + 100
        hidden_nodes = hidden_nodes + 1

        print(learning_rate)
        print(epochs)
        print(hidden_nodes)

        currAccuracy = run_backprop(dataset, k, learning_rate, epochs, hidden_nodes)

        if (currAccuracy > best_accuracy):
            best_accuracy = currAccuracy
            best_learning_rate = learning_rate
            best_epoch = epochs
            best_hidden_nodes = hidden_nodes

    return best_accuracy, best_learning_rate, best_epoch, best_hidden_nodes





    


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, learning_rate, epochs, n_outputs):
    for epoch in range(epochs):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)


# Initialize a network
def initialize_network(n_inputs, hidden_nodes, n_outputs):
    network = []
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(hidden_nodes)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(hidden_nodes + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, learning_rate, epochs, hidden_nodes):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, hidden_nodes, n_outputs)
    train_network(network, train, learning_rate, epochs, n_outputs)
    predictions = []
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)


def main():
    dataset = handle_data('iris.data')

    best_accuracy, best_learning_rate, best_epoch, best_hidden_nodes = tune_backprop(dataset, 5)
    print("Best Accuracy: ", best_accuracy)
    print("Best Learning Rate: ", best_learning_rate)
    print("Best epochs: ", best_epoch)
    print("Best Hidden Nodes: ", best_hidden_nodes)

main()
