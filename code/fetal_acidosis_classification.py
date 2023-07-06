# -*- coding: utf-8 -*-
"""
    Dummy template for classification
    B(E)3M33UI - Support script for the first semester task

    Jiri Spilka, 2019
"""
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import utils
import classifiers as clf 
import time

CSV_CTU = "Features_CTU_stat_spectral_figo_mfhurst_20190329.csv"
CSV_LYON = "NOT_AVAILABLE.csv"


our_scorer = make_scorer(utils.g_mean_score, greater_is_better=True)

def train_model(X, y):
    """
    Return a trained model.

    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert "X" in locals().keys()
    assert "y" in locals().keys()
    assert len(locals().keys()) == 2

    # create pipeline containing Support Vector Machine Model for acidosis classification
    pipe = clf.tune_svm_classifier(our_scorer)  

    # fit the model to the training data
    pipe.fit(X,y)
    return pipe

def predict(model1, X):
    """
    Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2
    
    return model1.predict(X)


def train_on_provided_data_test_on_external_data():
    """The model be tested using this script.
    External dataset will be used as a test data
    """

    # select labor stage and number of recording segments
    select_stage = 2  # CAN BE CHANGED
    nr_seg = 8  # CAN BE CHANGED

    # load training (provided data)
    df = utils.load_data_binary(CSV_CTU)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    # preprocessing (if necessary)
    x_train, y_train, _ = utils.get_X_y_from_dataframe(df)
    # drop rows containing NaN values
    X, y = utils.drop_nan_values(X,y)

    df = utils.load_data_binary(CSV_LYON)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    # preprocessing (if necessary)
    x_test, y_test, _ = utils.get_X_y_from_dataframe(df)
    # drop rows containing Nan values
    X_test, y_test = utils.drop_nan_values(X,y)

    print("\nTraining data CTU")
    print(f"y == 0: {sum(y_train == 0)}")
    print(f"y == 1: {sum(y_train == 1)}")

    print("\nTest data LYON ")
    print(f"y == 0: {sum(y_test == 0)}")
    print(f"y == 1: {sum(y_test == 1)}")

    # Train the model
    filter1 = train_model(x_train, y_train)

    # Compute predictions for training data and report g-mean
    # Ideally replace this with cross-validation g-mean, i.e. run CV on the CTU data
    y_tr_pred = predict(filter1, x_train)
    print("\ng-mean on training data: ", utils.g_mean_score(y_train, y_tr_pred))

    # Compute predictions for testing data and report our g-mean
    y_tst_pred = predict(filter1, x_test)
    print("g-mean on test data: ", utils.g_mean_score(y_test, y_tst_pred))

def train_test_on_provided_data():
    """Demonstration of model training and testing using the provided data
    You can do whatever you want with the provided data.
    The most important things:
        1 - your results should be reproducible
        2 - small change in a training data should not lead to large change in results
    """
    # select labor stage and number of segments
    select_stage = 2
    nr_seg = 8
    # load data
    df = utils.load_data_binary(CSV_CTU)
    df = utils.load_data_stage_last_k_segments(df, select_stage=select_stage, nr_seg=nr_seg)
    df = utils.df_drop_features(df)
    X, y, _ = utils.get_X_y_from_dataframe(df)

    # preprocessing 
    # drop rows containing NaN values
    X, y = utils.drop_nan_values(X,y)
    
    # split the available dataset into training and testing subsets
    test_data_size = 0.25
    train_data_size = 0.75    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, train_size = train_data_size)

    print("\nTraining data")
    print(f"y == 0: {sum(y_train == 0)}")
    print(f"y == 1: {sum(y_train == 1)}")

    print("\nTest data")
    print(f"y == 0: {sum(y_test == 0)}")
    print(f"y == 1: {sum(y_test == 1)}")

    # create and train model
    filter1 = train_model(X_train, y_train) 

    # predict the calsses of testing data
    y_tr_predict = predict(filter1, X_train)
    print("\ng-mean on training data: ", utils.g_mean_score(y_train, y_tr_predict))

    # Compute predictions for testing data and report our g-mean
    y_tst_pred = predict(filter1, X_test)
    print("g-mean on test data: ", utils.g_mean_score(y_test, y_tst_pred))

if __name__ == "__main__":  

    # create and use model on testing data - for testing
    timenow = time.time()
    train_test_on_provided_data()
    print(time.time() - timenow)
    # create and use model on external data - for evaluation
    #train_on_provided_data_test_on_external_data()
