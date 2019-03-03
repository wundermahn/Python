###################
# Author: Tony Kelly
# Date: February 2019
# Purpose: Script to clean data for 605.649 Intro to ML Course
# Description: This script looks through a csv file and replaces
# ? with the value in the last column (classification column).
##################

import csv
rows = []
with open('house.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    counter = 0

    for row in reader:

        for i, column in enumerate(row):
            if column == '?':
                row[i] = row[-1]
        rows.append(row)

with open('house.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')

    for row in rows:
        writer.writerow(row)