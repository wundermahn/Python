# Numpy Used for data handling
import numpy
# Statistics used for functions like mean and standard deviation
import statistics
# Math used for functions like the square root, exponentials, power functions, etc.
import math
# CSV used for handling non-numpy arrays
import csv
# Random used for randomizing arrays and dataset
import random

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
    randomize_data(data)
    return data


# Function that utilizes the numpy library to randomize the dataset.
def randomize_data(csv):
    csv = numpy.random.shuffle(csv)
    return csv


# Function to import a csv file into a normal, non numpy array
def import_normal_csv(file):
    # Create blank array
    results = []
    # Open file
    with open(file) as csvfile:
        # read in file changing values to floats
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            results.append(row)
    return results
#################### DATA HANDLING LIBRARY ####################

# This functions finds k
# Given a dataset, it utilizes numpys processing power to create an array of just the unique elements in a given column (i.e. classifications)
# The length of that array (# of unique things) is returned
def find_k(dataset):
    # Create an array of the last column in the dataset (classification column)
    arr = [dataset[j][-1] for j in range(len(dataset))]
    # Create an array of just the unique items in that array
    k = numpy.unique(arr)
    # Return the length of the unique items
    return len(k)


# This function calculates euclidean distance
# It relies on numpys data handling to avoid loops
# Capable of handling lists (clusters), individual points, etc.
def euclidean_distance(a, b):
    return numpy.sqrt(numpy.sum((a - b) ** 2))


# This function gets the "error" between centroids, which is
# the distance between old and new centroids
def get_centroid_error(centroid, old_centroid):
    # The total error
    total_error = 0

    # Determine the total error for each "column" of the centroid and add them together
    # Utilizing the Euclidean Distance function
    for index in range(len(centroid)):
        total_error = total_error + euclidean_distance(centroid[index], old_centroid[index])

    return total_error


# This function attempts to find the closest centroid to a given datapoint
def get_closest_centroid(datapoint, centroids):
    # Placeholder declaration for the best centroid
    bestCentroid = None
    # An arbitrary starting distance of a very large number
    bestDistance = 1000000000000

    # Loop through the centroids
    for currCent in centroids:
        # Calculate the current distance from the given datapoint and the current centroid
        currDistance = euclidean_distance(datapoint, currCent)
        # If it is the closest, update that
        if currDistance < bestDistance:
            bestDistance = currDistance
            bestCentroid = currCent
    # Return the closest
    return bestCentroid


# Function is intended to return the new centroid
# Which will be the mean of the points passed into it
def get_new_centroid(dataset, points):
    # Declare variables
    num_of_rows = len(points)
    num_of_columns = len(points[0])
    new_centroid = numpy.zeros(dataset.shape[1])

    # Loop through columns
    for index in range(num_of_columns):
        # Finding sum of columns to calculate mean
        sum = 0
        for point in points:
            # Go through all data points and add their values
            sum = sum + point[index]
        # Determine new centroid values by the total value of the individual points / the total # of instances
        new_centroid[index] = sum / num_of_rows

    # Return new centroid
    return new_centroid


# Function to return all points for a given cluster to which that points belongs
def get_same_cluster_points(points, cluster_labels, cluster):
    # Array of points we are going to return
    return_points = []

    # Loop through the labels
    for index in range(len(cluster_labels)):
        # If the label is equal to the cluster, add it
        if numpy.array_equal(cluster_labels[index], cluster):
            return_points.append(points[index])
    # Return the proper points
    return return_points


# Functions to return
def get_other_cluster_points(points, cluster_labels, cluster):
    # Array of points we are going to return
    return_points = []

    # Loop through the labels
    for index in range(len(cluster_labels)):
        # If the current label is the SAME as the given cluster, skip it
        if numpy.array_equal(cluster_labels[index], cluster):
            continue
        # If it is not, add it
        else:
            return_points.append(points[index])
    # Return the proper points
    return return_points


# Find the a sub i for the silhouette coefficient
def a_sub_i(points, single_point, cluster_labels, cluster):
    # Grab the points that belong to the same cluster as the given datapoint
    currPoints = get_same_cluster_points(points, cluster_labels, cluster)
    total = 0

    # Grab the Euclidean Distances from that point and all other points
    for point in currPoints:
        total = total + euclidean_distance(single_point, point)

    # Calculate the total
    total = total / len(currPoints)

    return total


# Find the b sub i for the silhouette coefficient
def b_sub_i(points, single_point, cluster_labels, cluster):
    # Determine all clusters by utilizing numpy's unique fucntionality
    all_clusters = numpy.unique(cluster_labels, axis=0)
    # Set minimum distance to something huge
    min_distance = 1000000000000000000000

    # Loop through all of the clusters
    # Look for clusters that are NOT equal to the given cluster of point x sub i
    for currCluster in all_clusters:
        currTotal = 0
        # If its the same, ship
        if numpy.array_equal(currCluster, cluster):
            continue
        # Else...
        else:
            # Get the points in the cluster
            currClusterPoints = get_same_cluster_points(points, cluster_labels, currCluster)
            # Run through each of the points and calculate the distance
            # Add that to total
            for point in currClusterPoints:
                currTotal = currTotal + euclidean_distance(single_point, point)
            # Try catch in the event it is 0
            try:
                currTotal = currTotal / len(currClusterPoints)
            except ZeroDivisionError:
                currTotal = 0
            # Find minimum distance
            if currTotal < min_distance:
                min_distance = currTotal

    return min_distance


# Find the silhouette coefficient for a single data point
def single_point_sil_coe(single_point, points, cluster_labels, cluster):
    # Variables for the numerator and denominator for the equation
    numerator = 0
    denominator = 0
    # Calculate a sub i and b sub i
    ai = a_sub_i(points, single_point, cluster_labels, cluster)
    bi = b_sub_i(points, single_point, cluster_labels, cluster)

    # Build the numerator
    numerator = bi - ai

    # Determine the denominator by finding the min of a sub i and b sub i
    if (ai) > (bi):
        denominator = ai
    else:
        denominator = bi

    # Return the silhouette coefficient
    return numerator / denominator


# Finds the total silhouette coefficient for the entire dataset
def total_sil_coe(points, cluster_labels):
    total = 0

    # Loop through all the points
    for index in range(len(points)):
        # For the current point and cluster, find the single point silhoeuette coefficient
        currPoint = points[index]
        currCluster = cluster_labels[index]
        total = total + single_point_sil_coe(currPoint, points, cluster_labels, currCluster)

    return total / len(points)


# Total K means function
# Commented well within
def k_means(dataset, k):
    # Turn the provided data into a numpy array
    dataset = numpy.asarray(dataset)
    # grab the initial centroid values
    # look through the numpy array, and find centroid values that look "like" the values in the dataset
    # this means finding the min and max and taking a random value in between
    centroids = dataset[numpy.random.choice(dataset.shape[0], k, replace=False), :]
    # cluster_labels are the initial labels to which every single datapoint in the dataset belongs, initially setting all to 0
    cluster_labels = numpy.zeros(dataset.shape)

    # # Calculate the current error of the centroids (distance between new and old)
    # for index in centroids
    # Starting with an error of 1, arbirtrary, but just to make sure the loop kicks off
    centroid_error = 1

    # While the centroids are still moving / function is still improving
    while centroid_error != 0:

        # Assign each value to its closest cluster_label
        for index in range(len(dataset)):
            # Assign each row to its closest centroids
            cluster_labels[index] = get_closest_centroid(dataset[index], centroids)

        # Loop through all of your current centroids
        for currCent in centroids:
            # Keep track of the centroid when we started to later calculate error
            oldCent = currCent
            # Create local array to store which data points currently belong to each cluster
            cluster_points = []
            # Loop through all of your current cluster cluster labels
            # Note that this should be the same size as your datasheet
            for index in range(len(cluster_labels)):
                # If the cluster label is the same thing as your current centroid
                if numpy.array_equal(cluster_labels[index], currCent):
                    # Add it to that local array
                    cluster_points.append(dataset[index])

            # If somehow there are NO associated points, the centroid is set to 0 to give
            # it another chance at getting some points
            if not cluster_points:
                currCent = numpy.zeros(currCent.shape)
            # If it did have points, get the new value of the centroid
            else:
                currCent = get_new_centroid(dataset, cluster_points)

            centroid_error = centroid_error + get_centroid_error(currCent, oldCent)

    return total_sil_coe(dataset, cluster_labels)


# Helper function to build the array for forward feature selection to use
def build_arr_to_test(arr_of_cols, arr_of_data):
    # Create blank array to return
    return_array = []
    # Loop through each row of the data
    for row in arr_of_data:
        data_row = []
        # Add the appropriate columns to the arrays
        for col_index in arr_of_cols:
            data_row.append(row[col_index])
        return_array.append(data_row)

    # Return the appropriate array with correct data to test
    return return_array


def stp_fwd_selc(array_of_data_points, k, numcolumns):
    # Create an array of columns to potentially evaluate
    # This array will hold the indices of the arrays to evaluate
    columns_in_pot = []
    count = 0
    # Loop through and append indices to the array based on the # of the columns in the entered dataset
    for index in range(0, numcolumns-1):
         columns_in_pot.append(count)
         count = count + 1
    # Shuffle the column order to avoid always starting at 0 and moving sequentially
    random.shuffle(columns_in_pot)
    # Columns chosen so far
    best_columns_to_use = []
    # Base Performance, set to a large small number
    base_performance = -10000000000000

    # While there are still features/columns to test
    while len(columns_in_pot) != 0:
        # Best performance for this run
        performance_so_far = -100
        # Best feature for this run
        best_col_so_far = None
        for col in columns_in_pot:
            # Test the column/feature by adding it to the list we will build our data from
            best_columns_to_use.append(col)
            # Build the data to pass to k_means
            data_to_test = build_arr_to_test(best_columns_to_use, array_of_data_points)
            # Get the performance of k_means with our selected data
            local_performance = k_means(data_to_test, k)
            print("CURRENTLY TESTING FEATURE[S]: " + str(best_columns_to_use) + " , ITS PERFORMANCE: " + str(
                local_performance))
            # Remove the column we are testing from our master list
            best_columns_to_use.remove(col)
            # If the performance was better than what we have seen this run
            if local_performance > performance_so_far:
                best_col_so_far = col
                performance_so_far = local_performance
        # After we have tested all of the columns for this run
        # Check to see if the performance increased at all
        if performance_so_far > base_performance:
            # We will add the best one to our master list of columns
            best_columns_to_use.append(best_col_so_far)
            # Remove the column from the pot
            columns_in_pot.remove(best_col_so_far)
            # Set new performance
            base_performance = performance_so_far
        else:
            # If you did not improve this run, end!!!!
            break
    # Print statements for visual output
    print("Your best columns were: ", best_columns_to_use)
    print("Your best performance was: ", base_performance)
    return (best_columns_to_use, base_performance)

# Main function, commented within
def main():
    print(" BEGINNING PROGRAM ")

    # Begin work on IRIS dataset
    print(" Working IRIS Dataset... ")
    # Convert the iris dataset to numpy array
    iris_dataset = csv_to_array('iris.csv')
    # find the k value for that dataset
    iris_k = find_k(iris_dataset)
    # Get a list of features (excluding the classification column)
    iris_list_of_features = numpy.asarray(iris_dataset.tolist()[:-1])
    # Create a raw data array without numpy
    iris_raw_data = import_normal_csv('iris.csv')
    # Shuffle the array
    random.shuffle(iris_raw_data)
    # Get the # of columns
    iris_numcolumns = iris_list_of_features.shape[1]

    # Run SFS w/ K Means
    stp_fwd_selc(iris_raw_data, iris_k, iris_numcolumns)

    print("             ")
    print("             ")

    # Begin work on GLASS dataset
    print(" Working GLASS Dataset... ")
    # Convert the glass dataset to numpy array
    glass_dataset = csv_to_array('glass.csv')
    # find the k value for that dataset
    glass_k = find_k(glass_dataset)
    # Get a list of features (excluding the classification column)
    glass_list_of_features = numpy.asarray(glass_dataset.tolist()[:-1])
    # Create a raw data array without numpy
    glass_raw_data = import_normal_csv('glass.csv')
    # Shuffle the array
    random.shuffle(glass_raw_data)
    # Get the # of columns
    glass_numcolumns = glass_list_of_features.shape[1]

    # Run SFS w/ K Means
    stp_fwd_selc(glass_raw_data, glass_k, glass_numcolumns)

    print("             ")
    print("             ")

    # Begin work on SPAM dataset
    print(" Working SPAM Dataset... ")
    # Convert the spam dataset to numpy array
    spam_dataset = csv_to_array('spam.csv')
    # find the k value for that dataset
    spam_k = find_k(spam_dataset)
    # Get a list of features (excluding the classification column)
    spam_list_of_features = numpy.asarray(spam_dataset.tolist()[:-1])
    # Create a raw data array without numpy
    spam_raw_data = import_normal_csv('spam.csv')
    # Randomize the data
    random.shuffle(spam_raw_data)
    # Get the # of columns
    spam_numcolumns = spam_list_of_features.shape[1]
    # Run SFS w/ K Means
    stp_fwd_selc(spam_raw_data, spam_k, spam_numcolumns)

main()