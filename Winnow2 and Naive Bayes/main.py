# Numpy Used for data handling
import numpy
# Statistics used for functions like mean and standard deviation
import statistics
# Math used for functions like the power function, needed to calculate exponentials
import math

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
    split_mark = int(num_rows / 3)

    # Split the data
    training_set = csv[:split_mark]
    test_set = csv[split_mark:]

    # Return the two data sets
    return training_set, test_set
#################### DATA HANDLING LIBRARY ####################

#################### Winnow2 Algorithm ####################
def winnow2(training_set, testing_set):
    # Define theta
    theta = 1
    # Define alpha
    alpha = 2
#### This portion of the algorithm deals with training ####
    # Define weights
    # Weights should be the same size (# of cols) as the training data, minus one, since we don't have a weight
    # for the classification column.
    weight_set = numpy.ones(training_set.shape[1] - 1)

    for row in training_set:
        # Define f(x), h(x), and the index for looping through the weight set
        fx = 0
        hx = 0
        weight_index = 0

        # Apply weights to each feature value, and then determine f(x) for the row
        # This should NOT apply to the classification column
        for column in row[:-1]:
            fx = fx + (column * weight_set[weight_index])
            weight_index = weight_index + 1

        # Determine hx by comparing to theta
        # This essentially determines your `"guess"
        # If you guessed correctly, increment your counter, and move on
        # If not, perform the promotion or demotion process
        if fx >= theta:
            hx = 1
        else:
            hx = 0

        if hx != row[-1]:
            # If you guessed 0 and needed a 1, you will need to perform promotion
            if hx == 0 and row[:-1] == 1:
                counter = 0
                # Check to see which feature values were 1
                # AKA loop through the columns in the row that might potentially need promotion
                # If they were, adjust the weights accordingly
                for check in row[:-1]:
                    if check == 1:
                        weight_set[counter] *= alpha
                        counter = counter + 1
            # If you guessed 1 and needed a 0, you will need to perform demotion
            if hx == 1 and row[:-1] == 0:
                counter = 0
                # Check to see which feature values were 1
                # AKA loop through the columns in the row that might potentially need demotion
                # If they were, adjust the weights accordingly
                for check in row[:-1]:
                    if check == 1:
                        weight_set[counter] /= alpha
                        counter = counter + 1

#### This portion of the algorithm deals with testing ####

    correct_guesses = 0
    for row in testing_set:
        # Define f(x), h(x), and the index for looping through the weight set
        fx = 0
        hx = 0
        weight_index = 0

        # Apply weights to each feature value, and then determine f(x) for the row
        # This should NOT apply to the classification column
        for column in row[:-1]:
            fx = fx + (column * weight_set[weight_index])
            weight_index = weight_index + 1

        # Determine hx by comparing to theta
        # This essentially determines your "guess"
        # If you guessed correctly, increment your counter, and move on
        # If not, perform the promotion or demotion process

        if fx >= theta:
            hx = 1
        else:
            hx = 0

        if (hx == row[-1]):
            correct_guesses = correct_guesses + 1
        # Print statements removed for demonstration
         #   print("CORRECT \n")

#        else:
#            print("INCORRECT \n")

    winnow_accuracy = 100 * (correct_guesses) / (testing_set.shape[0])
    return winnow_accuracy

#################### Winnow2 Algorithm ####################

#################### Naive Bayes Algorithm ####################

# In order to properly implement the NB classifier algorithm, a number of mathematical functions and, to a certain
# extent, data handling functions need to be written. Once that is done, we can then predict classifications in the nb_main
# functionality.

# First thing to do: Find out which feature values belong to each class. Essentially, create a map
# of feature values to classes.

def separate_classes(input_data):
    # Create a list of classifications
    # We are only doing binary in this case
    classifications = {}

    # create a variable for your loop
    sizeof = range(len(input_data))

    # Loop through the input data.
    # Add the classifications (last column) to the list if it does not already appear
    for index in sizeof:
        currRow = input_data[index]
        if (currRow[-1] not in classifications):
            classifications[currRow[-1]] = []
            classifications[currRow[-1]].append(currRow)
    return classifications

# The second thing to do: Determine what the means and standard deviations are, for each attribute, for each class
# To achieve this, we wil utilize another list with a zip function, which allows for value grouping
# This will produce an iterable list for each data row
def summarize_features(input_data):
    # Create the list
    feature_summaries = [(statistics.mean(value), statistics.pstdev(value)) for value in zip(*input_data)]
    # Delete the classification column
    del feature_summaries[-1]
    return feature_summaries

# The next thing we need to do is now group those summarized features into summarized features by class
# This is similar to the bucketing portion of the 2B module video

def summarize_features_by_class(input_data):
    # Gather the input data's classes
    classes = separate_classes(input_data)
    # Create a list of buckets to group the features by class
    buckets = {}
    # Create the features by class buckets
    for theClass, value in classes.items():
        buckets[theClass] = summarize_features(value)
    return buckets

# The next step is to begin calculating probabilities. The first of which is the
# Gaussian Probability Density. The goal would be to be able to etermine the probability
# of a given feature value for a given class. Again, classes here are only 0 and 1
def GaussianProbabilityDensity(feature_value, mean, standard_deviation):
    if (2*math.pow(standard_deviation, 2)) == 0:
        return 1
    # Declare a value for the Gaussian Probability Density
    GPD = 0
    # Determine the exponential needed for the calculation
    exponential = numpy.exp(-(numpy.pow(feature_value-mean, 2) / (2*numpy.pow(standard_deviation, 2))))
    # Calculate and return the GPD as a result of running the function
    GPD = (1 / (math.sqrt(2*math.pi)*standard_deviation) * exponential)
    return GPD

# The next step is to try and classify the data, not just determine its GPD
# This is done by determining the probabilities that a certain dataset of features belongs to a class
# Then, the function will try to find the largest probability and use that to create its first classification attempt
def Classify(feature_summaries, classifications):
    # Declare variables needed for the function
    # Including a list of possibilities, and the best Guess (classification) and its associated probability
    probabilities = {}
    bestGuess, bestProbability = None, -1
    # Loop through the data to classify and gather associated probabilities
    for classes, class_summaries in feature_summaries.items():
        probabilities[classes] = 1
        sizeof = range(len(class_summaries))
        # Calculate the probabilities
        for index in sizeof:
            mean, standard_deviation = class_summaries[index]
            feature_value = classifications[index]
            probabilities[classes] = probabilities[classes] * GaussianProbabilityDensity(feature_value, mean, standard_deviation)

    # Determine the BEST guess at what class the data instance belongs to
    for classification, probability in probabilities.items():
        if bestGuess is None or probability > bestProbability:
            bestProbability = probability
            bestGuess = classification
    return bestGuess

# Given a set of data, return the associated predictions
# Basically a cleaner function for Classify
def getPredict(results, testData):
    # Create a list of possible predictions
    predictions = []
    sizeof = range(len(testData))
    # Loop through the test data set and "classify" each row based on the learning above
    for index in sizeof:
        classification = Classify(results, testData[index])
        predictions.append(classification)
    return predictions

# A simple function to return the accuracy of the NB predictions
def getAccuracy(testData, predictions):
    correct = 0
    sizeof = range(len(testData))
    for index in sizeof:
        if testData[index][-1] == predictions[index]:
            correct = correct + 1
    return float((correct / int(len(testData))) * 100)

# A program to return the accuracy of the naive bayes implementation above
def naivebayes(training_set, testing_set):
    feature_summaries_by_class = summarize_features_by_class(training_set)
    predictions = getPredict(feature_summaries_by_class, testing_set)
    accuracy = getAccuracy(testing_set, predictions)
    return accuracy

#################### Naive Bayes Algorithm ####################

# Main function to display runtime output of algorithm. More detail to be discussed within.
def main():
    # Declare variables for use throughout the driver program.
    # Average accuracies to contain cumulative average accuracies for each algorithm
    # Counts is the number of iterations (i.e. times the algorithms were run)
    winnow_avg_accuracy = 0
    nb_avg_accuracy = 0
    counts = 0

    print ("Welcome to Winnow2 vs Naive Bayes Comparisons")
    print ("                                             ")

    ### Breast Cancer Data Set ###
    print("NOW CLASSIFYING: BREAST CANCER DATA ")

    # Load in the Breast Cancer Data
    myBDData = csv_to_array('bd.csv')
    # Split into testing and training sets
    testing_set, training_set = split_data(myBDData)

    # Run through 1000 iterations of running the winno2 and naivebayes algorithms
    for x in range(0,999):
        counts = counts + 1
        winnow_avg_accuracy += winnow2(training_set, testing_set)
        nb_avg_accuracy += naivebayes(training_set, testing_set)

    # Print out their accuracies
    print("Breast Cancer Data Winnow Average Accuracy: " + str(float(winnow_avg_accuracy) / counts))
    print("Breast Cancer Data Naive Bayes Average Accuracy : " + str(float(nb_avg_accuracy) / counts) + '\n')

    ### Iris Data Set ###
    print("NOW CLASSIFYING: IRIS DATA ")

    # Load in the Iris Data
    myIrisData = csv_to_array('iris.csv')
    # Split into testing and training sets
    testing_set, training_set = split_data(myIrisData)

    # Run through 1000 iterations of running the winnow2 and naivebayes algorithms
    for x in range(0, 999):
        counts = counts + 1
        winnow_avg_accuracy += winnow2(training_set, testing_set)
        nb_avg_accuracy += naivebayes(training_set, testing_set)

    # Print out their accuracies
    print("Iris Winnow Average Accuracy: " + str(float(winnow_avg_accuracy) / counts))
    print("Iris Naive Bayes Average Accuracy : " + str(float(nb_avg_accuracy) / counts) + '\n')

    ### House Votes Data Set ###
    print("NOW CLASSIFYING: HOUSE VOTE DATA ")

    # Load in the House Votes Data Set
    # Split into testing and training sets
    myHouseData = csv_to_array('house.csv')
    testing_set, training_set = split_data(myHouseData)

    # Run through 1000 iterations of running the winnow2 and naivebayes algorithms
    for x in range(0,999):
        counts = counts + 1
        winnow_avg_accuracy += winnow2(training_set, testing_set)
        nb_avg_accuracy += naivebayes(training_set, testing_set)

    # Print out the total accuracy averages for each algorithm
    print("Total Winnow Average Accuracy: " + str(float(winnow_avg_accuracy) / counts))
    print("Total Naive Bayes Average Accuracy : " + str(float(nb_avg_accuracy) / counts) + '\n')

### Extra Code
### After discussing with Professor Sheppard, unable to re-edit these data sets in time for submission
    # print(" NOW CLASSIFYING: SOYBEAN DATA ")
    #
    # mySBData = csv_to_array('sb.csv')
    # testing_set, training_set = split_data(mySBData)
    # 
    # for x in range(0, 999):
    #     counts = counts + 1
    #     winnow_avg_accuracy += winnow2(training_set, testing_set)
    #     nb_avg_accuracy += naivebayes(training_set, testing_set)
    #
    # print("SB Winnow Average Accuracy: " + str(float(winnow_avg_accuracy) / counts))
    # print("SB Naive Bayes Average Accuracy : " + str(float(nb_avg_accuracy) / counts) + '\n')
    #
    # print(" NOW CLASSIFYING: GLASS DATA ")
    #
    # myGlassData = csv_to_array('glass.csv')
    # testing_set, training_set = split_data(myGlassData)
    #
    # for x in range(0, 999):
    #     counts = counts + 1
    #     winnow_avg_accuracy += winnow2(training_set, testing_set)
    #     nb_avg_accuracy += naivebayes(training_set, testing_set)
    #
    # print("Total Winnow Average Accuracy: " + str(float(winnow_avg_accuracy)))
    # print("Total Naive Bayes Average Accuracy : " + str(float(nb_avg_accuracy)) + '\n')


main()