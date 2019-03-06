# Numpy Used for data handling
import numpy
# Statistics used for functions like mean and standard deviation
import statistics
# Math used for functions like the power function, needed to calculate exponentials
import math
# //TODO: ADD COMMENT
import operator
# //TODO: ADD COMMENT
from collections import Counter
# //TODO: ADD COMMENT
from numpy import genfromtxt

#################### DATA HANDLING LIBRARY ####################
def csv_to_array(file):
    # Open the file, and load it in delimiting on the ',' for a comma separated value file
    data = open(file, 'r')
    data = numpy.loadtxt(data, delimiter=',')

    # Loop through the data in the array
    for index in range(len(data)):
        # Utilize a try catch to try and convert to float, if it can't convert to float, converts to 0
        try:
            data[index] = [float(x) for x in data[index]]
        except Exception:
            data[index] = 0
        except ValueError:
            data[index] = 0

    # Return the now type-formatted data
    return data

def machine_csv_to_array(file):
    data = genfromtxt(file, delimiter=',')
    return data

# Function that utilizes the numpy library to randomize the dataset.
def randomize_data(csv):
    csv = numpy.random.shuffle(csv)
    return csv

# Function to split the data into test and training sets
# 67% of the data is used to train, while 33% is used to test
def split_data(csv):
    # Call the randomize data function
    randomize_data(csv)
    # Grab the number of rows and calculate where to split
    num_rows = csv.shape[0]
    split_mark = int(num_rows / 3)

    # Split the data
    train_set = csv[:split_mark]
    test_set = csv[split_mark:]

    # Split the data into classes vs actual data
    training_cols = train_set.shape[1]
    testing_cols = test_set.shape[1]
    training_classes = train_set[:,training_cols-1]
    testing_classes = test_set[:,testing_cols-1]

    # Take the training and testing sets and remove the last (classification) column
    training_set = train_set[:-1]
    testing_set = test_set[:-1]

    # Return the datasets
    return testing_set, testing_classes, training_set, training_classes

#################### DATA HANDLING LIBRARY ####################

def euclidean_distance(a, b):
    return numpy.sqrt(numpy.sum((a - b) ** 2))

# This function returns the most element from an array
# This will be used to return the classification for the k-nearest-neighbor
def get_classification(classes):
    count = Counter(classes)
    return count.most_common()[0][0]

def get_average_value(values):
    values = numpy.asarray(values)
    average = numpy.average(values)
    return average

def squared_error(guess, test_point):
    error = guess - test_point
    error = error * error
    return error

def get_nearest_neighbors(training_data, evaluation_point, k):
    distances = []
    index = 0

    for neighbor in training_data:
        currDistance = euclidean_distance(neighbor, evaluation_point)
        distances.append((index, currDistance))
        index = index + 1

    distances.sort(key=operator.itemgetter(1))

    nearest_neighbors = []
    for jindex in range(k):
        nearest_neighbors.append(distances[jindex][0])

    return nearest_neighbors

def known_nearest_neighbors_classification(testing_data, testing_classes, training_data, training_classes, k):
    num_correct = 0
    num_wrong = 0
    total = 0
    for test_class, test_point in zip(testing_classes, testing_data):
        neighbor_indeces = get_nearest_neighbors(training_data, test_point, k)
        nearest_neighbors = []

        for index in neighbor_indeces:
            nearest_neighbors.append(training_classes[index])

        classification = get_classification(nearest_neighbors)

        if numpy.array_equal(classification, test_class):
            num_correct = num_correct + 1
        else:
            num_wrong = num_wrong + 1

    total = float(((num_correct) / (num_correct + num_wrong)) * 100)
    return total

def known_nearest_neighbors_regression(testing_data, testing_classes, training_data, training_classes, k):
    errors = []

    for test_class, test_point in zip(testing_classes, testing_data):
        neighbor_indeces = get_nearest_neighbors(training_data, test_point, k)
        nearest_neighbors = []

        for index in neighbor_indeces:
            nearest_neighbors.append(training_classes[index])

        average = get_average_value(nearest_neighbors)
        error = squared_error(average, test_point)
        errors.append(error)

    return (get_average_value(errors))

def main():
    ecoli_csv_data = csv_to_array('Classification/ecoli.csv')
    ecoli_testing_data, ecoli_testing_classes, ecoli_training_data, ecoli_training_classes = split_data(ecoli_csv_data)

    seg_csv_data = csv_to_array('Classification/segmentation.csv')
    seg_testing_data, seg_testing_classes, seg_training_data, seg_training_classes = split_data(seg_csv_data)


    print("NOW TESTING SEGMENTATION")
    seg_accuracy = known_nearest_neighbors_classification(seg_testing_data, seg_testing_classes, seg_training_data, seg_training_classes, 3)
    print("                      ")
    print("NOW TESTING ECOLI")
    ecoli_accuracy = known_nearest_neighbors_classification(ecoli_testing_data, ecoli_testing_classes, ecoli_training_data, ecoli_training_classes, 3)

    print("Segmentation Accuracy: ", seg_accuracy)
    print("Ecoli Accuracy: ", ecoli_accuracy)

    print("#########################################################")
    print("#########################################################")
    print("#########################################################")

    fire_csv_data = csv_to_array('Regression/forestfires.csv')
    fire_testing_data, fire_testing_classes, fire_training_data, fire_training_classes = split_data(fire_csv_data)

    machines_csv_data = csv_to_array('Regression/machine.csv')
    machine_testing_data, machine_testing_classes, machine_training_data, machine_training_classes = split_data(machines_csv_data)

    fires_error = known_nearest_neighbors_regression(fire_testing_data, fire_testing_classes, fire_training_data, fire_training_classes, 3)
    machines_error = known_nearest_neighbors_regression(machine_testing_data, machine_testing_classes, machine_training_data, machine_training_classes, 3)

    print("Fires Error: ", fires_error)
    print("Machines Error: ", machines_error)

main()
