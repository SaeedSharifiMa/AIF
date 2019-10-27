# documentation
# script for: Synthetic Data Generation

import numpy as np
import pandas as pd

############### Inputs ################
num_individuals = 2000
dimension = 20
num_problems = 75
red_proportion = 0.75 # proportion of red people: imagine there are "red" and and "blue" people
coin = 0.8 # >= 0.5 to favor red people, the probability that a red person falls into the majority group of each task
            # a blue persion falls into a majority group with probability 1 - coin
random_seed = 123456
#######################################

np.random.seed(random_seed)
features = 2*np.random.randint(2, size=(num_individuals, dimension))-1
weights_majority = 2*np.random.randint(2, size=(num_problems, dimension))-1
weights_minority = 2*np.random.randint(2, size=(num_problems, dimension))-1
labels = np.zeros(shape=(num_individuals, num_problems))

for i in range(num_individuals):
    for j in range(num_problems):
        if i <= int(red_proportion*num_individuals): # red person
            if np.random.binomial(n=1, p=coin, size=1)[0] == 1:
                labels[i,j] = np.sign( np.dot(weights_majority[j,], features[i,]) )
            else:
                labels[i,j] = np.sign( np.dot(weights_minority[j,], features[i,]) )
        else: # blue person
            if np.random.binomial(n=1, p=coin, size=1)[0] == 0:
                labels[i,j] = np.sign( np.dot(weights_majority[j,], features[i,]) )
            else:
                labels[i,j] = np.sign( np.dot(weights_minority[j,], features[i,]) )

labels = (0.5*(1 + labels)).astype(int)
col_names = [ ("problem_%i" %i) for i in range(num_problems) ]
labels_dataset = pd.DataFrame(data = labels, columns = col_names)
features_dataset = pd.DataFrame(data = features)
dataset = pd.concat([features_dataset, labels_dataset], axis = 1)
dataset.to_csv('dataset/synthetic.csv')
