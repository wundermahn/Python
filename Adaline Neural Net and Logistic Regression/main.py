# numpy Used for data handling and some mathematical calculations
import numpy
# statistics used for functions like mean and standard deviation
import statistics
# math used for functions like the power function, needed to calculate exponentials
import math
# sys used to take in arguments from the end user / command line
import sys
# Used for data frame handling
import pandas as pd

#################### DATA HANDLING LIBRARY ####################
# This functions converts the passed csv file to an array
def csv_to_array(file):
    # Open the file, and load it in delimiting on the ',' for a comma separated value file
    data = open(file, 'r')
    # Use numpy
    data = numpy.loadtxt(data, delimiter=',')

    # Loop through the data in the array
    for index in range(len(data)):
        # Utilize a try catch to try and convert to float, if it can't convert to float, converts to 0
        try:
            data[index] = [float(x) for x in data[index]]
        except ValueError:
            data[index] = 0

    # Return the now type-formatted data
    return data


# Function that utilizes the numpy library to randomize the dataset.
def randomize_data(csv):
    # Shuffle the dataset
    csv = numpy.random.shuffle(csv)
    # Return it
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
    training_set = csv[:split_mark]
    test_set = csv[split_mark:]

    # Return the two data sets
    return training_set, test_set


# Function to split the data into test and training sets
# 67% of the data is used to train, while 33% is used to test
def split_log_data(csv):
    # Call the randomize data function
    randomize_data(csv)
    # Grab the number of rows and calculate where to split
    num_rows = csv.shape[0]
    split_mark = int(num_rows * .67)

    # Split the data
    train_set = csv[:split_mark]
    test_set = csv[split_mark:]

    # Split the data into classes vs actual data
    training_classes = train_set[:, -1]
    testing_classes = test_set[:, -1]

    # Take the training and testing sets and remove the last (classification) column
    training_set = train_set[:, :-1]
    testing_set = test_set[:, :-1]
    # Return the datasets
    return testing_set, testing_classes, training_set, training_classes


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
        # Divide the # of occurences by total rows
        weight = float((total / num_total_rows))
        # Add that weight to the list of class weights
        class_weights.append(weight)

    # Turn the weights into a numpy array
    class_weights = numpy.asarray(class_weights)
    # Return the array
    return classes, class_weights


# This functions performs 5 cross fold validation for classification
def cross_fold_validation_classification(dataset, k):
    # Grab the classes and class weights (their occurence distributions) across the dataset
    classes, class_weights = class_distribution(dataset)
    # Grab the total length of the dataset
    total_num_rows = dataset.shape[0]
    # Create a copy of the dataset to actually manipulate
    data = numpy.copy(dataset)

    # Create an array with hold the 5 folds
    total_fold_array = []

    # Basic for loop to iterate 5 times
    for _ in range(k):
        # Set an array of the current fold being created
        curr_fold_array = []

        # Loop through each class and its associated weight
        for a_class, a_class_weight in zip(classes, class_weights):
            # s the remaining data
            randomize_data(data)
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
    # Cast the folds as a numpy array
    total_fold_array = numpy.asarray(total_fold_array)

    # Return the folds
    return total_fold_array

# This function performs one hot encoding on the classes
def one_hot_encode(data):
    # Turn the classes, or passed data, into a series
    series = pd.Series(list(data))
    # Perform one hot encoding using the pandas library get_dummies function
    one_hot_data = pd.get_dummies(series)
    # Return the one_hot_encoded data
    return one_hot_data

#################### DATA HANDLING LIBRARY ####################
#################### Naive Bayes Algorithm ####################

# Create a map of data to classes
def separate_classes(data):
    # Create a list of classifications
    # We are only doing binary in this case
    classifications = {}

    # create a variable for your loop
    sizeof = range(len(data))

    # Loop through the input data.
    # Add the classifications (last column) to the list if it does not already appear
    for index in sizeof:
        # Set the currRow to to be the current row
        currRow = data[index]
        # If the current classification is not in the list of classifications, append it
        if (currRow[-1] not in classifications):
            classifications[currRow[-1]] = []
            classifications[currRow[-1]].append(currRow)
    return classifications


# Determine what the means and standard deviations are, for each attribute, for each class
# To achieve this, we wil utilize another list with a zip function, which allows for value grouping
# This will produce an iterable list for each data row
def summarize_features(data):
    # Create the list
    feature_summaries = [(statistics.mean(value), statistics.pstdev(value)) for value in zip(*data)]
    # Delete the classification column
    del feature_summaries[-1]
    # Return the groupings
    return feature_summaries


# The next thing we need to do is now group those summarized features into summarized features by class
# This is similar to the bucketing portion of the 2B module video
def summarize_features_by_class(data):
    # Gather the input data's classes
    classes = separate_classes(data)
    # Create a list of buckets to group the features by class
    buckets = {}
    # Create the features by class buckets
    for theClass, value in classes.items():
        buckets[theClass] = summarize_features(value)
    return buckets


# The next step is to begin calculating probabilities. The first of which is the
# Gaussian Probability Density. The goal would be to be able to determine the probability
# of a given feature value for a given class.
def GaussianProbabilityDensity(feature_value, mean, standard_deviation):
    if (2 * math.pow(standard_deviation, 2)) == 0:
        return 1
    # Declare a value for the Gaussian Probability Density
    GPD = 0
    # Determine the exponential needed for the calculation
    exponential_val = numpy.exponential(-(numpy.pow(feature_value - mean, 2) / (2 * numpy.pow(standard_deviation, 2))))
    # Calculate and return the GPD as a result of running the function
    GPD = (1 / (math.sqrt(2 * math.pi) * standard_deviation) * exponential_val)
    # Return the GPD
    return GPD


# Determine the probabilities that a certain dataset of features belongs to a class
# Then, the function will try to find the largest probability and use that to create its first classification attempt
def classify(feature_summaries, classifications):
    # Declare variables needed for the function
    # Including a list of possibilities, and the best Guess (classification) and its associated probability
    probabilities = {}
    # Set arbitrary bestGuess and bestProbability
    bestGuess, bestProbability = None, -1

    # Loop through the data to classify and gather associated probabilities
    for classes, class_summaries in feature_summaries.items():
        probabilities[classes] = 1
        # Create sizeof variable for loop
        sizeof = range(len(class_summaries))
        # Calculate the probabilities
        for index in sizeof:
            # Grab mean and standard deviation
            mean, standard_deviation = class_summaries[index]
            # Grab currennt feature value
            feature_value = classifications[index]
            # Determine the probabilities
            probabilities[classes] = probabilities[classes] * GaussianProbabilityDensity(feature_value, mean,
                                                                                         standard_deviation)

    # Determine the BEST guess at what class the data instance belongs to
    for classification, probability in probabilities.items():
        # If it is better than the best
        if probability > bestProbability:
            # Set the best probability to this probability
            bestProbability = probability
            # Set the best guess to this guess
            bestGuess = classification

    # Return the best guess
    return bestGuess


# Given a set of data, return the associated predictions
# Basically a cleaner function for Classify
def get_naivebayes_prediction(results, testData):
    # Create a list of possible predictions
    predictions = []
    # Create sizeof for the loop
    sizeof = range(len(testData))
    # Loop through the test data set and "classify" each row based on the learning above
    for index in sizeof:
        # Grab the classification
        classification = classify(results, testData[index])
        # Append to the predictions array
        predictions.append(classification)

    # Return the predictions
    return predictions


# A simple function to return the accuracy of the NB predictions
def get_naivebayes_accuracy(testData, predictions):
    # Number of correct guesses
    correct = 0
    # Create sizeof for loop
    sizeof = range(len(testData))
    # Loop through the testing data and see if you guessed right
    for index in sizeof:
        # If you did
        if testData[index][-1] == predictions[index]:
            # Increase the # of correct
            correct = correct + 1
        # If not, continue
        else:
            continue

    # Return your accuracy percentage
    return float((correct / int(len(testData))) * 100)


# A program to return the accuracy of the naive bayes implementation above
def naivebayes(training_set, testing_set):
    # Summarize the features by class
    feature_summaries_by_class = summarize_features_by_class(training_set)
    # Grab your predictions
    predictions = get_naivebayes_prediction(feature_summaries_by_class, testing_set)
    # Grab your accuracy by seeing how well you guessed
    accuracy = get_naivebayes_accuracy(testing_set, predictions)
    # Return that accuracy
    return float(accuracy)


######################## Naive Bayes Algorithm ########################
#################### Logistic Regression Algorithm ####################

# This function determines if a set of classes are multiclass
# This is used to determine which of the sigmoid or softmax functions should be used
def isMulticlass(classes):
    # Create a temp numpy array of the data passed in
    temp = numpy.asarray(classes)
    # create an array of the unique classes
    unique_classes = numpy.unique(temp)
    # If there are more than 2 possible options, it is multiclass
    if (len(unique_classes) > 2):
        return True
    # IF there are 2 or less (impossible for this assignment), return false, meaning it is a binary class problem
    else:
        return False


# This function calculates the exponentials of normalized data
def exponential(num):
    # Set the max of the entered number
    max = num.max()
    # Set the y value to be the exponential of the entered number minus its max
    y = numpy.exp(num - max)
    # Return that value over its sum
    return y / y.sum()


# Function to calculate the sigmoid value of a given value
# Used for 2-class problems
def sigmoid(val):
    # Calculate Z
    Z = 1 / (1 + exponential(-val))
    # Return Z
    return Z.astype(float)


# Function to calculate the softmax value of
# Used for multi class problems
def softmax(val):
    # Calculate Z
    Z = exponential(val) / float(sum(exponential(val)))
    # Return Z
    return Z.astype(float)


# Function that returns the log likelihood for the given data input
def log_likelihood(features, target, weights):
    # Get the dot product using the numpy library
    scores = numpy.dot(features, weights)
    # Determine the current likelihood for the given set of data
    currLikelihood = numpy.sum(target * scores - (numpy.log(1 + numpy.exponential(scores))))
    # Return the current likelihood
    return currLikelihood


# This function formats the feature data into horizontal stacks so the dot product can be calculated
def format_feature_data(features):
    # Align the values with a blank numpy array of all ones
    alignment = numpy.ones((features.shape[0], 1))
    # Now rearrange the features appropriately based off of the numpy array
    # Essentially mapping the feature vector to a single column
    new_features = numpy.hstack((alignment, features))

    # Return the new features
    return new_features


# This function runs the logistic regression algorithm for the given dataset
def logistic_regression(training_features, training_classes, learning_factor):
    # Determine if the problem is a multiclass or a binary class problem
    multiclass = isMulticlass(training_classes)

    # If there are more than 2 classes, perform one hot encoding
    if (multiclass == True):
        new_classes = one_hot_encode(training_classes)

    # Reformat the training feature data
    # This is where we are essentially getting the feature data in a one hot encoded format
    new_features = format_feature_data(training_features)

    # Create a blank numpy array to hold the weights, which is an array of zeroes
    # that is the same size as the set of features
    weights = numpy.zeros(new_features.shape[1])

    # Loop through the num_steps to take, which is a tuning function
    # These are the number of iterations to make. This is an adjustable "threshold"
    # 25000 was selected to result in good testing of the data.
    for step in range(50000):
        # Grab your current scores
        scores = numpy.dot(new_features, weights)

        # If there are 2 classes, use the sigmoid function
        if multiclass == False:
            predictions = sigmoid(scores)
        # If there are more than 2 classes, use the softmax function
        else:
            predictions = softmax(scores)

        # Grab the current error
        currError = training_classes - predictions
        # Determine your current gradient via transposing your features and the current error
        gradient = numpy.dot(new_features.T, currError)
        # Add this to your weights array
        weights = weights + ( learning_factor * gradient)

    # Return the weights
    return weights


# This function gets the accuracy of the logistic regression model
def get_logisticregression_predictions(weights, testing_features, testing_classes):
    # Determine if the problem is multiclass
    multiclass = isMulticlass(testing_classes)

    # If there are more than 2 classes, perform one hot encoding
    if (multiclass == True):
        new_classes = one_hot_encode(testing_classes)

    # Reformat the testing feature data
    new_features = format_feature_data(testing_features)
    # Get the scores, which are the dot product of the data and the weights that were passed in
    final_scores = numpy.dot(new_features, weights)

    # Determine if the classes are more than 2 and if so use the proper math function
    if multiclass == False:
        # Test the accuracy by using the sigmoid function
        predictions = numpy.round(sigmoid(final_scores))
    else:
        # Test the accuracy by using the softmax function
        predictions = numpy.round(softmax(final_scores))

    # Return the predictions
    return predictions

def get_logisticregression_accuracy(testing_classes, predictions):
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


#################### Logistic Regression Algorithm ####################
############################## Adaline Network ########################

# An OO-type structure was utilzied for the adaline network due to the network itself being an object to which other objects
# (like data) must be passed through. The same structure that was used in Programming Assignment 4 was repeated in this instance
class adaline_network(object):
    # Initiate a new network
    def __init__(self, learning_rate, num_iterations, weights_initialized):
        # Set its learning rate, the number of iterations to use, and if the weights are already initialized
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights_initialized = weights_initialized

    # This allows us to delete the network to clear memory for a new one
    def __del__(self):
        # Define the delete function for this class instance
        del self

    # Use the dot product to calculate the net input
    def net_input(self, data):
        # Calculate it using the numpy library
        input = numpy.dot(data, self.w[1:]) + self.w[0]
        # Return it
        return input

    # This function calculates the activation function for the given data passed in
    def activation_function(self, data):
        # Calculate the activation function
        threshold = self.net_input(data)

        # Return the activation function
        return threshold

    # This aligns the training data in the network and returns the network with the training data loaded in
    def fit_data(self, data, classes):
        # Initialize the weights
        self.initialize_weights(data.shape[1])
        # Set the cost of the neural net to be an empty list at the moment
        self.cost = []
        # Loop through the number of iterations that were set for this neural net
        for i in range(self.num_iterations):
            # Set a temp cost list to use
            temp_cost = []
            # For x sub i in the zipped list of the training data and classes
            for x_sub_i, target in zip(data, classes):
                # Append the updated weights to the cost function
                temp_cost.append(self.update_weights(x_sub_i, target))
            # Determine the average cost
            avg_cost = sum(temp_cost) / len(classes)
            # Lastly, move the "average" cost from our temp_cost list to be the cost for the neural net
            self.cost.append(avg_cost)

        # Return the neural net
        return self

    # This initializes the weight values for the neural net
    def initialize_weights(self, data_shape):
        # Initial weights are randomized using the numpy library
        self.w = numpy.random.normal(loc=0.0, scale=1.0, size = 1 + data_shape)
        # Set the weights_initialized attribute of the class to be true
        self.weights_initialized = True

    # This function updates weights if they have already been applied
    def update_weights(self, x_sub_i, actual_class):
        # Your output should be the activation function of x sub i
        # This is a class
        output = self.activation_function(x_sub_i)
        # Your error is the target value minus the output
        # The difference in the class
        error_rate = calculate_error(self, actual_class, output)

        # Adjust the weights as described in module 10C
        self.w[1:] = self.w[1:] + (self.learning_rate * x_sub_i.dot(error_rate))
        self.w[0] = self.w[0] + (self.learning_rate * error_rate)

        # Adjust your cost based on the error
        new_cost = ((error_rate / 2) ** 2)

        # Return the cost
        return new_cost

    # This function actually grabs the predictions
    def get_adaline_prediction(self, data):
        # Create an empty list of predictions
        predictions = []
        # Loop through the data that was passed in
        for item in data:
            # If the activation function is greater or equal to 0, then append a 1
            if (self.activation_function(item) >= 0):
                predictions.append(1)
            # If not
            else:
                # Append a 0
                predictions.append(0)

        # Return the predictions
        return predictions

# This function returns the error rate for the neural net
def calculate_error(self, target, value):
    # Calculate the distance from the target value to the current value
    error = (target - value)
    # Return the error
    return error

# This function, though separate from the class, calculates the accuracy of the adaline network's guesses
def get_adaline_accuracy(vals, predictions):
    # Set a variable to collect the # of correct guesses
    num_correct = 0
    # Set a temp predictions array to convert the predictions to numpy
    temp_predictions = numpy.asarray(predictions)
    # Now reconvert that to a float array
    predictions = temp_predictions.astype(numpy.float)

    # Loop through each of the actual values
    for index in range(len(vals)):
        # If the actual value is the same as your prediction / guess
        if (vals[index] == predictions[index]):
            # Increase the number of correct
            num_correct = num_correct + 1
        # Otherwise
        else:
            # Continue on
            continue

    # Return a float of the num correct / total values
    return float(num_correct / len(vals))


# This function is used to run an instance of an adaline network
def run_adaline(training_data, training_classes, testing_data, testing_classes, learning_rate, num_iterations,
                weights_initialized):
    # Create a new instantiation
    currNetwork = adaline_network(learning_rate, num_iterations, weights_initialized)
    # Fit the network with the current instance of data
    currNetwork.fit_data(training_data, training_classes)
    # Grab the predictions from the network (attempts at guessing classes)
    predictions = currNetwork.get_adaline_prediction(testing_data)

    # Delete the current network
    del currNetwork

    # Return the predictions
    return predictions


# This function is used to tune the adaline network to find the optimal number of epochs and the best learning rate
def tune_run_adaline(testing_data, testing_classes, training_data, training_classes):
    # Declare variables for the learning rates, number of steps to use, and if the weights for the network have already been ininitalized
    learning_rate = 0
    num_iterations = 0
    best_accuracy = 0
    weights_initialized = False

    # Basic for loop with 50 iterations
    # Moves the learning rate from 0.0 -> 0.50
    # Moves the epoch count from 0.0 -> 50.0
    for _ in range(50):
        # Grab the current predictions from the algorithm
        predictions = run_adaline(testing_data, testing_classes, training_data, training_classes, learning_rate,
                                  num_iterations, weights_initialized)

        # Calculate the accuracy
        accuracy = get_adaline_accuracy(testing_classes, predictions) * 100

        # If your accuracy is better than the best
        if accuracy > best_accuracy:
            # Set the best accuracy to the current
            best_accuracy = accuracy
        # If not
        else:
            # Continue
            continue

        # Increase the learning rate by .01
        learning_rate = learning_rate + .01
        # Increase the epoch count by 1
        num_iterations = num_iterations + 1
        # Set the weights_initialized boolean to true
        weights_initialized = True

    # Return the best accuracy
    return best_accuracy


############################## Adaline Network ########################

# This function runs the algorithms for this project, and is intended as a helper function to make the main function cleaner
def run_algorithms(name, dataset, k, verbose):
    # Print the name of the dataset being passed in as a string
    print("                   ")
    print("NOW TESTING:", name)

    # Set "best" overall accuracies for the naive bayes (nb), logistic regression (lr), and adaline network (a) runs
    best_nb_accuracy = 0
    best_lr_accuracy = 0
    best_a_accuracy = 0

    # Create your k number of folds, where k is the variable passed in
    folds = cross_fold_validation_classification(dataset, k)
    # Keep track of the # of folds, used mainly for printing purposes
    num_folds = 0

    # Loop through the folds
    for fold in folds:
        # Create a training and testing set for naive bayes
        nb_training_set, nb_testing_set = split_data(fold)
        # Create a training and testing set for logistic regression, where the classes are separated
        lr_testing, lr_testing_classes, lr_training, lr_training_classes = split_log_data(fold)

        # Set the current naive bayes accuracy by running the algorithm on the current training and testing sets created from the fold
        curr_nb_accuracy = naivebayes(nb_training_set, nb_testing_set)
        # Grab the current weights for the logistic regression algorithm
        lr_weights = logistic_regression(lr_training, lr_training_classes, 0.01)
        # Grab the current predictions for the logistic regressino algorithm
        lr_predictions = get_logisticregression_predictions(lr_weights, lr_testing, lr_testing_classes)
        # Calculate the current logistic regression accuracy by running the algorithm on the current training and testing sets created from the fold
        curr_lr_accuracy = get_logisticregression_accuracy(lr_testing_classes, lr_predictions)

        # Create a training and testing set for adaline network
        a_testing, a_testing_classes, a_training, a_training_classes = split_log_data(fold)
        # Calculate the current accuracy for the adaline network
        curr_a_accuracy = tune_run_adaline(a_testing, a_testing_classes, a_training, a_training_classes)

        # If the current naive bayes accuracy is better than the best
        if (curr_nb_accuracy > best_nb_accuracy):
            # Set the best as the current
            best_nb_accuracy = curr_nb_accuracy

        # If the current logistic regression accuracy is better than the last
        if (curr_lr_accuracy > best_lr_accuracy):
            # Set the best as the current
            best_lr_accuracy = curr_lr_accuracy

        # If the current adaline network accuracy is better than the last
        if (curr_a_accuracy > best_a_accuracy):
            # Set the best as the current
            best_a_accuracy = curr_a_accuracy

        # Increase the number of folds, mainly used for printing purposes
        num_folds = num_folds + 1

        # If verbosity is enabled (via the 1), then print out stuff for each fold
        if (verbose == 1):
            print("FOLD NUM: ", num_folds, "Naive Bayes: ", '%.2f' % curr_nb_accuracy, "Logistic Regression: ",
                  '%.2f' % curr_lr_accuracy,
                  "Adaline Network: ", '%.2f' % curr_a_accuracy)

    # Print out the overall best results for the algorithms
    print("For", name, "the best Naive Bayes Accuracy was: ", '%.2f' % best_nb_accuracy,
          " and the best Logistic Regression Accuracy was: ", '%.2f' % best_lr_accuracy,
          " and the best Adaline Accuracy was: ", '%.2f' % best_a_accuracy)


# Main function that is well commented within
def main():
    # Load in all of the datasets as commandline arguments into numpy arrays
    breast_cancer_array = csv_to_array(sys.argv[1])
    glass_array = csv_to_array(sys.argv[2])
    house_array = csv_to_array(sys.argv[3])
    iris_array = csv_to_array(sys.argv[4])
    soybean_array = csv_to_array(sys.argv[5])

    # Run all of the algorithms using the run_algorithms function created to clean the code
    # 5 is passed in as the k for k-cross-fold-validation
    # 1 is used to enable the verbosity function. Can be switched to 0 for one-liner print statements per run
    run_algorithms("Breast Cancer", breast_cancer_array, 5, 1)
    run_algorithms("Glass", glass_array, 5, 1)
    run_algorithms("House Votes", house_array, 5, 1)
    run_algorithms("Iris", iris_array, 5, 1)
    run_algorithms("Soybean", soybean_array, 5, 1)


main()
