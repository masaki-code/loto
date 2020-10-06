import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '../file/trial_mini.csv'

import csv

with open(path) as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

print('===============')

with open(path) as f:
    reader = csv.reader(f)
    l = [row for row in reader]

print(l)
