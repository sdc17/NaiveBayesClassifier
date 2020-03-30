import os

# Seed for random
seed = input('seed(e.g. 2017013632): ')
if not seed:
    seed = '2017013632'

# Divide the training set by k parts
command_kflod = 'python tools.py --k=5 --seed={}'.format(seed)


# Training
command_naivebayes = 'python NaiveBayes.py --k=5 --sample=1.0 --rank=0.2 --fuzzy=0.25 --seed={}'.format(seed)


# Predict 
command_predict = 'python Predict.py --alpha=1e-30 --record=False --seed={}'.format(seed)

os.system(command_kflod)
os.system(command_naivebayes)
os.system(command_predict)
