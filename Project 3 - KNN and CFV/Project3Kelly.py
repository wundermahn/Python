# Numpy Used for data handling
import numpy
# Operator used for returning data uniformly
import operator
# Counter object used to count occurences
from collections import Counter
# Used to pass arguments to the program
import sys

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
    split_mark = int(num_rows * .8)

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

# This functions returns the euclidean distance between two points
def euclidean_distance(a, b):
    # This function utilizes the numpy library which allows us to return the distance between two points, or
    # between two matrices (numpy arrays)
    return numpy.sqrt(numpy.sum((a - b) ** 2))

# This function returns the most element from an array
# This will be used to return the classification for the k-nearest-neighbor
def get_classification(classes):
    # Utilize the Counter library to count the number of instances of each item in the array
    count = Counter(classes)
    # Utilize the Counter library to return the element of the array with the highest number of instances
    return count.most_common()[0][0]

# This function gets the average value of an array
def get_average_value(values):
    # Convert the given array to a numpy array
    values = numpy.asarray(values)
    # Create a value to hold the average of the numpy array
    average = numpy.average(values)
    # Return the average of the given array
    return average

# This functions calculates the mean squared error between the data values passed to it
def mean_squared_error(averages, points):
    # Convert the given two arrays to numpy arrays
    averages = numpy.asarray(averages)
    points = numpy.asarray(points)

    # This function utilizes the numpy library which allows us to return the mean squared error between two points,
    # or between two matrices (numpy arrays)
    total = (numpy.square(averages-points)).mean()
    return total

# This function grabs the k nearest neighbors from the training data from the given evaluation points
def get_nearest_neighbors(training_data, evaluation_point, k):
    # Create an array to hold the distances to each point in the training data from the given evaluation point
    distances = []
    # Create an index to return the proper points from the training data
    index = 0

    # Loop through the training data
    for neighbor in training_data:
        # Calculate the current euclidean distance from the current point to the given evaluation point
        currDistance = euclidean_distance(neighbor, evaluation_point)
        # Add that point to the distances array
        distances.append((index, currDistance))
        # Increment the index
        index = index + 1

    # Once the loop is done, sort the distances array
    distances.sort(key=operator.itemgetter(1))

    # Create an array of nearest neighbors
    nearest_neighbors = []
    # Loop through "k", or the number of nearest neighbors to retrieve
    for jindex in range(k):
        # Add them to the nearest neighbors array
        nearest_neighbors.append(distances[jindex][0])

    # Return the nearest neighbors array
    return nearest_neighbors
# This function implements the known nearest neighbors algorithm for classification purposes
# This function is well commented within
def known_nearest_neighbors_classification(testing_data, testing_classes, training_data, training_classes, k):
    # Variable declarations for the number of guesses correct, incorrect, and total
    num_correct = 0
    num_wrong = 0
    total = 0

    # Loop through the test classes and the test data
    for test_class, test_point in zip(testing_classes, testing_data):
        # Create an array of the indices of the number of nearest neighbors
        neighbor_indeces = get_nearest_neighbors(training_data, test_point, k)
        # Create an array for the actual nearest neighbors
        nearest_neighbors = []

        # Loop through the indices of the nearest neighbors, and grab the corresponding training data for each index
        for index in neighbor_indeces:
            # Append the training data to the nearest_neighbors array
            nearest_neighbors.append(training_classes[index])

        # Determine the classification by getting the most commonly appearing value out of the array
        # i.e., return the majority classification
        classification = get_classification(nearest_neighbors)

        # If the current testing_point's class equals the attempted guess at classification...
        if numpy.array_equal(classification, test_class):
            # Increase the number of correct guesses
            num_correct = num_correct + 1
        # If it was an incorrect guess
        else:
            # Increase the number of incorrect guesses
            num_wrong = num_wrong + 1

    # Create the average accuracy of classification
    total = float(((num_correct) / (num_correct + num_wrong)) * 100)
    # Return the average accuracy
    return total

# This function implements the known nearest neighbors algorithm for regression purposes
# This function is well commented within
def known_nearest_neighbors_regression(testing_data, testing_classes, training_data, training_classes, k):
    # Create two arrays
    # One of averages (i.e., the average value of the nearest neighbors)
    # the other of the evaluation points
    # These will be used later to calculate mean squared error
    array_of_averages = []
    array_of_points = []

    # Loop through the testing data, and the testing "class"
    # class here really refers to the value we are predicting
    for test_class, test_point in zip(testing_classes, testing_data):
        # Create an array of the indices of the number of nearest neighbors
        neighbor_indeces = get_nearest_neighbors(training_data, test_point, k)
        # Create an array for the actual nearest neighbors
        nearest_neighbors = []

        # Loop through the indices of the nearest neighbors, and grab the corresponding training data for each index
        for index in neighbor_indeces:
            # Append the training data to the nearest_neighbors array
            nearest_neighbors.append(training_classes[index])
        # Create an average of the values of the nearest nieghbors
        average = get_average_value(nearest_neighbors)
        # Append this to the array of average values
        array_of_averages.append(average)
        # Append the current evaluation prediction point to the array of points
        array_of_points.append(test_point[-1])

    # Create the mean squared error
    mse = mean_squared_error(array_of_averages, array_of_points)
    # Return the mean squared error
    return (mse)

# This function performs the condensed nearest neighbors (cnn) algorithm
def condensed_nearest_neighbors(training_data, training_classes):
    # Get the total number of rows
    total_num_rows = training_data.shape[0]
    # Create array "z" as outlined in Alpaydin page 195
    z = []
    # Create the corresponding classes array for the datapoints in z
    z_classes = []
    # Instantiate the two arrays with some random points
    z.append(training_data[0])
    z_classes.append(training_classes[0])

    # Boolean value to see if Z has changedor not
    isChanged = True

    # Size of z to check
    size = len(z)

    while(isChanged == True):
        # Create an array of random numbers to be used for testing
        # -1 since we already used a row
        indices = list(range(0,total_num_rows-1))
        # Cast as a numpy array and randomize it
        indices = numpy.asarray(indices)
        numpy.random.shuffle(indices)

        # Loop through the random set of indices
        for index in indices:
            # Get the closest point
            closest_point = get_nearest_neighbors(z, training_data[index], 1)
            # Get the class
            closest_point_class = z_classes[closest_point[-1]]
            # Check to see if it was classified correctly
            if training_classes[index] == closest_point_class:
                continue
            # If it was not, add it to z and z_classes
            else:
                z.append(training_data[index])
                z_classes.append(training_classes[index])

        # Check to see if z has changed
        # If it has not, set the isChanged bool to False so the loop will break
        if len(z) == size:
            isChanged = False
        # If it HAS, then reset the size variable
        else:
            size = len(z)

    # Convert to numpy arrays and return z and the associated classes
    z = numpy.asarray(z)
    z_classes = numpy.asarray(z_classes)
    return z, z_classes

# This function is intended to optimize the known nearest neighbor runs by iterating through 50 runs of each algorithm
# with k moving from 1-20
def optimize_knn_classification(testing_data, testing_classes, training_data, training_classes):
    # Set the total amount of runs
    max_runs = 20
    # Set the running best accuracy, or most accurate run so far
    best_accuracy = 0
    # Set the k associated with the best run so far
    best_k =[]

    # Loop through the max runs
    for k in range(max_runs):
        # Increase K (we need to have at least one nearest neighbor)
        k=k+1
        # Set the current performance to the results of the classification
        currPerf = known_nearest_neighbors_classification(testing_data, testing_classes, training_data, training_classes, k)
        print("CURRENT K: ", k, "CURRENT PERFORMANCE: " ,currPerf)
        #If the current performance is equal to the best accuracy
        if currPerf == best_accuracy:
            # Append this value to the k array
            best_k.append(k)
        # If the current performance was better than the best so far
        if currPerf > best_accuracy:
            # Set the best equal to the current
            best_accuracy = currPerf
            # Reset the best k array, and append the current k to it
            best_k = []
            best_k.append(k)
        # If it was not, make sure to continue in the loop
        else:
            continue

    # Return the best accuracy and the best associated k
    return best_accuracy, best_k

# This function is intended to optimize the known nearest neighbor runs by iterating through 50 runs of each algorithm
# with k moving from 1-20
def optimize_cnn_classification(testing_data, testing_classes, training_data, training_classes):
    # Set the total amount of runs
    max_runs = 20
    # Set the running best accuracy, or most accurate run so far
    best_accuracy = 0
    # Set the k associated with the best run so far
    best_k =[]

    # Loop through the max runs
    for k in range(max_runs):
        # Increase K (we need to have at least one nearest neighbor)
        k=k+1
        # Set the current performance to the results of the classification
        currPerf = known_nearest_neighbors_classification(testing_data, testing_classes, training_data, training_classes, k)
        print("w/CNN | CURRENT K: ", k, "CURRENT PERFORMANCE: " ,currPerf)
        # If the current performance is equal to the best, add this k to the array:
        if currPerf == best_accuracy:
            best_k.append(k)
        # If the current performance was better than the best so far
        if currPerf > best_accuracy:
            # Set the best equal to the current
            best_accuracy = currPerf
            # Reset the best k array, and append the current k to it
            best_k = []
            best_k.append(k)
        # If it was not, make sure to continue in the loop
        else:
            continue

    # Return the best accuracy and the best associated k
    return best_accuracy, best_k

# This function is intended to optimize the known nearest neighbor runs by iterating through 50 runs of each algorithm
# with k moving from 1-20
def optimize_knn_regression(testing_data, testing_classes, training_data, training_classes):
    # Set the total amount of runs
    max_runs = 20
    # Set the running least_error, or best run so far
    # Make it something enormous so the runs are definitely better than it
    least_error = 1000000000000000
    # Set the k associated with the best run so far
    # It can be possible for more than 1 k value to give the same performance
    best_k = []

    # Loop through the max runs
    for k in range(max_runs):
        # Increase K (we need to have at least one nearest neighbor)
        k=k+1
        # Set the current performance to the results of this regression run
        currError = known_nearest_neighbors_regression(testing_data, testing_classes, training_data, training_classes, k)
        print("CURRENT K: ", k, "CURRENT PERFORMANCE: ", currError)
        #If this runs error equaled the least total, add that k value to the best k array
        if currError == least_error:
            best_k.append(k)
        # If this run produced a lower error than the lowest already
        if currError < least_error:
            # Set the least error to the current
            least_error = currError
            # Reset the best_k array and add the current k to it
            best_k = []
            best_k.append(k)
        # If not, continue the loop
        else:
            continue

    # Return the least error and the best k associated with it
    return least_error, best_k

# This function returns the list of classes, and their associated weights (i.e. distributions)
# for a given dataset
def class_distribution(dataset):
    # Ensure the dataset is a numpy array
    dataset = numpy.asarray(dataset)
    # Collect # of total rows and columns, using numpy
    num_total_rows = dataset.shape[0]
    num_columns = dataset.shape[1]
    # Create a numpy array of just the classes
    classes = dataset[:,num_columns-1]
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
        # Divide the # of occurences by total rows
        weight = float((total/num_total_rows))
        # Add that weight to the list of class weights
        class_weights.append(weight)

    # Turn the weights into a numpy array
    class_weights = numpy.asarray(class_weights)
    # Return the array
    return classes, class_weights

# This functions performs 5 cross fold validation for classification
def cross_fold_validation_classification(dataset):
    # Grab the classes and class weights (their occurence distributions) across thed ataset
    classes, class_weights = class_distribution(dataset)
    # Grab the total length of the dataset
    total_num_rows = dataset.shape[0]
    # Create a copy of the dataset to actually manipulate
    data = numpy.copy(dataset)

    # Create an array with hold the 5 folds
    total_fold_array = []

    # Basic for loop to iterate 5 gimes
    for _ in range(5):
        # Set an array of the current fold being created
        curr_fold_array = []

        # Loop through each class and its associated weight
        for a_class,a_class_weight in zip(classes, class_weights):
            # Shuffle the remaining data
            numpy.random.shuffle(data)
            # Keep track of how many items have been added
            num_added = 0
            # Determine hom many items to add, based on the distribution in the respective classes
            num_to_add = float((((a_class_weight * total_num_rows)) / 5))
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
    # Cast the folds as a numpy array
    total_fold_array = numpy.asarray(total_fold_array)

    # Return the folds
    return total_fold_array

# This function performs a modified cross fold validation for regression
def cross_fold_validation_regression(dataset):
    # Shuffle the dataset
    numpy.random.shuffle(dataset)
    # Set an array for all of the folds
    total_fold_array = []

    # Determine the number of total rows in the dataset, and the mark to properly split the data into fifths
    num_total_rows = dataset.shape[0]
    split_mark = int(num_total_rows / 5)

    # Create five "temp arrays", which are folds, using the appropriate splits
    temp_fold_one = dataset[:split_mark]
    temp_fold_two = dataset[split_mark:split_mark*2]
    temp_fold_three = dataset[split_mark*2:split_mark*3]
    temp_fold_four = dataset[split_mark*3:split_mark*4]
    temp_fold_five = dataset[split_mark*4:split_mark*5]

    # Merge all of the "temporary" fold arrays into the parent array
    total_fold_array.append(temp_fold_one)
    total_fold_array.append(temp_fold_two)
    total_fold_array.append(temp_fold_three)
    total_fold_array.append(temp_fold_four)
    total_fold_array.append(temp_fold_five)

    # Cast the array as a numpy array and return it
    total_fold_array = numpy.asarray(total_fold_array)
    return total_fold_array

# This is a wrapper function to properly run knn based on which type of knn (regression or classification) is passed
def run_knn(dataset, type):
    # 0 for classification, 1 for regression
    if type == 0:
        # Assumes classification. Set the best accuracy and k across all folds
        best_accuracy = 0
        best_k = []
        condensed_best_accuracy = 0
        condensed_best_k = []

        print("You are performing KNN on classification data.")

        ### NORMAL RUNS ###
        # Determine the folds
        folds = cross_fold_validation_classification(dataset)
        # Loop through each fold
        # Declare fold num to print out which fold you are on. For demonstration
        fold_num = 1
        for fold in folds:
            # Print current fold
            print("CURRENT FOLD: ", fold_num)
            # Make sure it is a numpy array (it should already be)
            fold = numpy.asarray(fold)
            # Create training and testing sets (and classes)
            # this also handles some data manipulation needed for the knn function
            testing_set, testing_classes, training_set, training_classes = split_data(fold)
            # Run KNN classification on the fold
            accuracy, k = optimize_knn_classification(testing_set, testing_classes, training_set, training_classes)
            # Increase the fold_num
            fold_num = fold_num + 1
            # If your current accuracy was better than the best for all folds
            if accuracy > best_accuracy:
                # Set it equal, and switch your best k
                best_accuracy = accuracy
                best_k = k
            # If not, continue
            else:
                continue
        ### NORMAL RUNS ###

        ### CONDENSED RUNS ###
        folds = cross_fold_validation_classification(dataset)
        # Loop through each fold
        # Declare fold num to print out which fold you are on. For demonstration
        fold_num = 1
        for fold in folds:
            # Print current fold
            print("CURRENT FOLD: ", fold_num)
            # Make sure it is a numpy array (it should already be)
            fold = numpy.asarray(fold)
            # Create training and testing sets (and classes)
            # this also handles some data manipulation needed for the knn function
            testing_set, testing_classes, training_set, training_classes = split_data(fold)
            # Run condensed nearest neighbors on the training data
            condensed_training_set, condensed_training_classes = condensed_nearest_neighbors(training_set, training_classes)
            # Run KNN classification on the fold
            condensed_accuracy, condensed_k = optimize_cnn_classification(testing_set, testing_classes, condensed_training_set, condensed_training_classes)
            # Increase the fold_num
            fold_num = fold_num + 1
            # If your current accuracy was better than the best for all folds
            if condensed_accuracy > condensed_best_accuracy:
                # Set it equal, and switch your best k
                condensed_best_accuracy = condensed_accuracy
                condensed_best_k = condensed_k
            # If not, continue
            else:
                continue
        ### CONDENSED RUNS ###

        # Print out the results
        print("Normal Best K :", best_k)
        print("Normal Best Accuracy :", best_accuracy)
        print("                      ")
        print("Condensed Best K: ", condensed_best_k)
        print("Condensed Best Accuracy :", condensed_best_accuracy)
        print("                      ")

    # If it was regression...
    else:
        # Set variables for the best error rate and k across all folds
        # Initiate the error as something enormous
        least_error = 100000000
        best_k = []

        print("You are performing KNN on regression data")

        # Determine the folds
        folds = cross_fold_validation_regression(dataset)
        # Loop through each fold
        # Declare fold num to print out which fold you are on. For demonstration
        fold_num = 1
        for fold in folds:
            # Print current fold
            print("CURRENT FOLD: ", fold_num)
            # Make sure it is a numpy array (it should already be)
            fold = numpy.asarray(fold)
            # Create training and testing sets (and classes)
            # this also handles some data manipulation needed for the knn function
            testing_set, testing_classes, training_set, training_classes = split_data(fold)
            # Run KNN regression on the fold
            error, k = optimize_knn_regression(testing_set, testing_classes, training_set, training_classes)
            # Increase the fold_num
            fold_num = fold_num + 1
            # If your error is "better" (less than) the best, set the least to be the current, and capture the k values
            if error < least_error:
                least_error = error
                best_k = k
            else:
                continue


        # Print out the results
        print("Best K: ", best_k)
        print("Least Error: ", least_error)
        print("                      ")

# Main function, this is well commented within
def main():
    print("BEGINNING PROGRAM")

    # Collect inputs from the user
    ecoli_data = sys.argv[1]
    segmentation_data = sys.argv[2]
    fire_data = sys.argv[3]
    machine_data = sys.argv[4]

    # Create datasets
    ecoli = csv_to_array(ecoli_data)
    segmentation = csv_to_array(segmentation_data)
    fires = csv_to_array(fire_data)
    machines = csv_to_array(machine_data)

    # Perform the experiment, and with proper print statements
    # As described in the run_knn function, provide 0 for classification or 1 for regression
    print("Now running: Ecoli")
    run_knn(ecoli,0)
    print("Now running: Segmentation")
    run_knn(segmentation,0)
    print("Now running: Forest Fires")
    run_knn(fires,1)
    print("Now running: Machines")
    run_knn(machines, 1)

    print("ENDING PROGRAM")

main()