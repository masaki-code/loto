import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '../file/trial_mini.csv'

with open(path) as f:
    s = f.read()

print(type(s))
print(s)
