# generalization over problems

import numpy as np
import pandas as pd
import classred as red
import clean_data as parser1
import pickle
from sklearn import linear_model


############################### LEARNING ORACLE ##############################
class RegressionLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        cost_vec0 = Y * W  # cost vector for predicting zero
        cost_vec1 = (1 - Y) * W # cost vector for predicting one
        self.reg0 = linear_model.LinearRegression()
        self.reg0.fit(X, cost_vec0)
        self.reg1 = linear_model.LinearRegression()
        self.reg1.fit(X, cost_vec1)
      
    def predict(self, X):
        pred0 = self.reg0.predict(X)
        pred1 = self.reg1.predict(X)
        return 1*(pred1 < pred0)
#############################################################################

#############################################################################
def binarize_attr(a):
    """
    given a set of attributes; binarize them
    """
    for col in a.columns:
        if len(a[col].unique()) > 2:  # hack: identify numeric features
            sens_mean = np.mean(a[col])
            a[col] = 1 * (a[col] > sens_mean)
##############################################################################

############### Inputs ################
dataset = "communities"
max_iter = 1000
nu = 0.001
alpha_list = [i/100 for i in range(5,28)] # the range of alpha values to plot
#######################################

x_training, x_test, y_training0, y_training, y_test0, y_test = parser1.clean_communities_test()

binarize_attr(y_training0)
binarize_attr(y_training)
binarize_attr(y_test0)
binarize_attr(y_test)

n_train = x_training.shape[0]
m_train = y_training0.shape[1]
n_test = x_test.shape[0]
m_test = y_test0.shape[1]

x_test = x_test.reset_index(drop=True)
y_training = y_training.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

training_err = []
training_unf = []
test_err_individuals = []
test_unf_individuals = []
test_err_problems = []
test_unf_problems = []
test_err_both = []
test_unf_both = []

for alpha in alpha_list:
    results = pickle.load( open("communities_pickles_largetrain/%(dataset)s_%(max_iter)i_%(alpha).2f_%(nu).3f_aif.p" %locals(), "rb") )
    weight_set = results['weight_set'] 
    weights = results['weights']
    T = weight_set.shape[1]
    training_err.append(results['err'])
    training_unf.append(results['unfairness_prime'])
    
    weighted_preds_individuals = pd.DataFrame(columns = y_training0.columns)
    for col in y_training0.columns:
        preds_individuals = pd.DataFrame()
        for t in range(T):
            learner = RegressionLearner()
            learner.fit(X = x_training, Y = y_training0[col], W = weight_set[t])
            preds_individuals[t] = learner.predict(x_test)
        weighted_preds_individuals[col] = preds_individuals[weights.index].dot(weights)

    weighted_preds_problems = pd.DataFrame(columns = y_test0.columns)
    weighted_preds_both = pd.DataFrame(columns = y_test.columns)
    for col in y_test.columns:
        preds_problems = pd.DataFrame()
        preds_both = pd.DataFrame()
        for t in range(T):
            learner = RegressionLearner()
            learner.fit(X = x_training, Y = y_test0[col], W = weight_set[t])
            preds_problems[t] = learner.predict(x_training)
            preds_both[t] = learner.predict(x_test)
        weighted_preds_problems[col] = preds_problems[weights.index].dot(weights)
        weighted_preds_both[col] = preds_both[weights.index].dot(weights)

    err_individual_individuals = {}
    for i in range(n_test):
        err_individual_individuals[i] = sum(np.abs(y_training.loc[i] - weighted_preds_individuals.loc[i])) / m_train
    
    err_individual_problems = {} 
    for i in range(n_train):
        err_individual_problems[i] = sum(np.abs(y_test0.loc[i] - weighted_preds_problems.loc[i])) / m_test

    err_individual_both = {}
    for i in range(n_test):
        err_individual_both[i] = sum(np.abs(y_test.loc[i] - weighted_preds_both.loc[i])) / m_test

    err_individuals = (y_training - weighted_preds_individuals).abs().values.mean()
    err_problems = (y_test0 - weighted_preds_problems).abs().values.mean()
    err_both = (y_test - weighted_preds_both).abs().values.mean()

    unf_individuals = max(err_individual_individuals.values()) - min(err_individual_individuals.values())
    unf_problems = max(err_individual_problems.values()) - min(err_individual_problems.values())
    unf_both = max(err_individual_both.values()) - min(err_individual_both.values())

    test_err_individuals.append(err_individuals)
    test_err_problems.append(err_problems)
    test_err_both.append(err_both)

    test_unf_individuals.append(unf_individuals)
    test_unf_problems.append(unf_problems)
    test_unf_both.append(unf_both)

output = {'training_err': training_err, 'test_err_individuals': test_err_individuals, 'test_err_problems': test_err_problems, 'test_err_both': test_err_both, 'training_unf': training_unf, 'test_unf_individuals': test_unf_individuals, 'test_unf_problems': test_unf_problems, 'test_unf_both': test_unf_both}
pickle.dump(output, open('generalization_pickles/generalization_%(dataset)s_%(n_train)i_%(m_train)i_%(n_test)i_%(m_test)i.p' %locals(), 'wb'))

