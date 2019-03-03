###################
# Author: Tony Kelly
# Date: February 2019
# Purpose: Script to clean data for 605.649 Intro to ML Course
# Description: This script looks through a csv file and changes values to binary 0 or 1.
# -> The script takes the average value of the respective value's column, and if the value is greater than the average,
# -> marks it as a 1, if not, marks it as a 0.
##################

import csv

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return 0

with open('glass.csv', 'r') as inf:
    cr = csv.reader(inf)
    sums = [safe_float(x) for x in next(cr)]
    n = 0
    for row in cr:
        float_row = [safe_float(x) for x in row]
        sums = [x + y for x, y in zip(sums, float_row)]
        n += 1
    averages = [x / n for x in sums]

with open('glass.csv', 'r') as inf:
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
