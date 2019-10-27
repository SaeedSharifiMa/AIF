# documentation
# output: clean data set
# for each data set 'name.csv' we create a function clean_name

import numpy as np
import pandas as pd


def center(X):
    for col in X.columns:
        X.loc[:, col] = X.loc[:, col]-np.mean(X.loc[:, col])
    return X


def add_intercept(X):
    """Add all 1's column to predictor matrix"""
    X['intercept'] = [1]*X.shape[0]



def clean_communities():
    """Clean communities & crime data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)
    #y = df['ViolentCrimesPerPop']
    #q_y = np.percentile(y, 70)
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    #y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
    X = df.iloc[:, 0:122]
    X = center(X)
    #X = add_intercept(X)
    n = X.shape[0]
    n_train = int(0.8*n)
    features = X.iloc[0:n_train,50:70]
    problems = X.iloc[0:n_train,0:50]
    return features, problems

def clean_communities_test():
    """Clean communities & crime data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)
    #y = df['ViolentCrimesPerPop']
    #q_y = np.percentile(y, 70)
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    #y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
    X = df.iloc[:, 0:122]
    X = center(X)
    #X = add_intercept(X)
    n = X.shape[0]
    n_train = int(0.8*n)

    x_training = X.iloc[0:n_train,50:70]
    x_test = X.iloc[n_train:n,50:70]

    y_training0 = X.iloc[0:n_train,0:50]
    y_training = X.iloc[n_train:n,0:50]
    y_test0 = X.iloc[0:n_train,70:95]
    y_test = X.iloc[n_train:n,70:95]
    return x_training, x_test, y_training0, y_training, y_test0, y_test 


def clean_synthetic():
    df = pd.read_csv('dataset/synthetic.csv')
    ind = list(df.columns).index('problem_0')
    end = len(list(df.columns))
    n = df.shape[0]
    n_train = int(0.8*n)
    m = end - ind
    m_train = int(2*m/3)

    features = df.iloc[0:n_train,0:ind]
    problems = df.iloc[0:n_train,ind:(ind+m_train)]
    return features, problems

def clean_synthetic_test():
    df = pd.read_csv('dataset/synthetic.csv')
    ind = list(df.columns).index('problem_0')
    end = len(list(df.columns))
    n = df.shape[0]
    n_train = int(0.8*n)
    m = end - ind
    m_train = int(2*m/3)

    x_training = df.iloc[0:n_train,0:ind]
    x_test = df.iloc[n_train:n,0:ind]

    y_training0 = df.iloc[0:n_train,ind:(ind+m_train)]
    y_training = df.iloc[n_train:n,ind:(ind+m_train)]
    y_test0 = df.iloc[0:n_train,(ind+m_train):end]
    y_test = df.iloc[n_train:n,(ind+m_train):end]

    return x_training, x_test, y_training0, y_training, y_test0, y_test

