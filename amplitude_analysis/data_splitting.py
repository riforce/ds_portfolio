# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:48:06 2023

@author: riley
"""

#useful imports
import numpy as np
import csv
import random
import sklearn as sk
from sklearn import model_selection
from pandas import *
import pandas as pd


def split_woodanderson(train_percent, test_percent, file):
    """
    Splits a file into testing and training subsets. 

    Parameters
    ----------
    train_percent : float
        Size of the training set, as a percent.
    test_percent : float
        Size of the testing set, as a percent.
    file : string
        Location of data file to split. Assumes a .csv file.

    Returns
    -------
    train : DataFrame
        The resultant training set.
    test : DataFrame
        The resultant testing set. 

    """
    #read csv into a dataframe and check the head
    df = pd.read_csv(file)

    #set a random seed - lucky number 13
    seed = 13
    random.seed(seed)

    #create a partition of training/testing events

    #start by getting the unique event IDs
    eventlist = df["evid"].unique()

    #With sklearn, use model_selection's train_test_split() function.
    #Use the seed as the random state, and shuffle
    train_samp, test_samp = sk.model_selection.train_test_split(eventlist, train_size = train_percent, test_size = test_percent, random_state = seed, shuffle = True)
    
    #make the train and test dataframes
    train = df[df['evid'].isin(train_samp)]
    test = df[df['evid'].isin(test_samp)]
    
    return train, test
    


if __name__ == "__main__":
    filename = "~/repos/seismology_data/mlWoodAnderson.csv"
    train_p = 0.8
    test_p = 0.2
    
    #call split_woodanderson
    train, test = split_woodanderson(train_p, test_p, filename)
    
    #Write the training and testing sets into new CSV files
    train.to_csv("~/repos/seismology_data/woodanderson_train.csv", sep = ",")
    test.to_csv("~/repos/seismology_data/woodanderson_test.csv", sep = ",")