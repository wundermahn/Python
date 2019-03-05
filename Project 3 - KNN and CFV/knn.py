# Numpy Used for data handling
import numpy
# Statistics used for functions like mean and standard deviation
import statistics
# Math used for functions like the power function, needed to calculate exponentials
import math
# //TODO: ADD COMMENT
import operator

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

def euclidean_distance(a, b):
    return numpy.sqrt(numpy.sum((a - b) ** 2))

# This function returns the most element from an array
# This will be used to return the classification for the k-nearest-neighbor
def get_classification(classes):
    # Create a list of counts for each class
    counts = {}

    # Loop through the given array in a for each loop
    for aclass in classes:
        if aclass[:,0] in counts:
            print(aclass[:,0])
            counts[aclass[:,0]] += 1
        else:
            counts[aclass[:,0]] = 1

    print(counts[aclass[:,0]])
    classification = sorted(counts, key=counts.get, reverse=True)
    print(classification[0])
    return classification[0]

def get_nearest_neighbors(training_data, evaluation_point, k):
    distances = []
    index = 0

    for neighbor in training_data:
        currDistance = euclidean_distance(neighbor, evaluation_point)
        distances.append(index, currDistance)
        index = index + 1

    distances.sort(key=operator.itemgetter(1))

    nearest_neighbors = []
    for jindex in range(k):
        nearest_neighbors.append(distances[jindex][0])

    return nearest_neighbors

def knn_predict(test_data, train_data, k_value):
    for i in test_data:
        eu_Distance =[]
        knn = []
        good = 0
        bad = 0
        for j in train_data:
            eu_dist = euclidean_distance(i, j)
            eu_Distance.append((j[5], eu_dist))
            eu_Distance.sort(key = operator.itemgetter(1))
            knn = eu_Distance[:k_value]
            for k in knn:
                if k[0] =='g':
                    good += 1
                else:
                    bad +=1
        if good > bad:
            i.append('g')
        elif good < bad:
            i.append('b')
        else:
            i.append('NaN')

#def split_by_class
# Create a function to essentially split into regular and just classes

def main():
    ecoli_csv_data = csv_to_array('Classification/ecoli.csv')
    ecoli_training_set, ecoli_testing_set = split_data(ecoli_csv_data)

    cols = ecoli_training_set.shape[1]
    ecoli_classes = ecoli_training_set[:,cols-1]
    print(ecoli_classes)
    print(len(ecoli_classes))
    print(len(ecoli_training_set))

main()
