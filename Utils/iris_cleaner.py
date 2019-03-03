###################
# Author: Tony Kelly
# Date: February 2019
# Purpose: Script to clean data for 605.649 Intro to ML Course
# Description: This script changes Iris-sentosa to 0, all else to 1, and then takes average
# -> of each column. It then compares the values contains within the columns and if they are above
# -> or equal to the average, marks it as 1, if not, marks it as 0.
##################

import csv

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return 0

with open('iris_data.csv', 'r') as inf:
    cr = csv.reader(inf)
    sums = [safe_float(x) for x in next(cr)]
    n = 0
    for row in cr:
        float_row = [safe_float(x) for x in row]
        sums = [x + y for x, y in zip(sums, float_row)]
        n += 1
    averages = [x / n for x in sums]

with open('iris_data.csv', 'r') as inf:
    with open('output.csv', 'w', newline='') as outf:
        cr = csv.reader(inf)
        cw = csv.writer(outf)
        for row in cr:
            float_row = [safe_float(x) for x in row]
            cw.writerow([1 if x > a else 0 for x, a in zip(float_row, averages)])

with open('output.csv', 'r', newline='') as dirty, open ('final_output.csv', 'w', newline='') as clean:
    reader = csv.reader(dirty)
    writer = csv.writer(clean)

    for row in reader:
        writer.writerow(row[:-1])
