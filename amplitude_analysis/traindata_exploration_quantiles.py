# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:24:38 2023

@author: riley
"""


#useful imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from patsy import dmatrices

#read in our training data
train_file = "~/repos/seismology_data/woodanderson_train.csv"
data = pd.read_csv(train_file)

y_train, x_train = dmatrices('diff_pick_ot_time ~ source_receiver_distance', data, return_type = 'dataframe')

def conditional_means_scatter():
    """ 
    Plots a scatterplot of the source receiver distance vs the diff pick ot time,
    represented in darker blue, as well as the conditional means, represented in 
    lighter blue and overlaid. 
    The plot includes the best fit line in red, calculated using OLS regression, and
    a line in green representing the fit line of the median model. 
    """
    #First let's plot the source receiver distance (srd) against diff pick ot time (dpot).
    fig = plt.figure()
    fig.suptitle('Signal Velocity & Conditional Means')
    plt.xlabel('SRD')
    plt.ylabel('DPOT')

    plt.scatter(x = data['source_receiver_distance'],
            y = data['diff_pick_ot_time'],
            label = "original", color = "darkblue", alpha = 0.8)

    srd = np.array(data.groupby('source_receiver_distance')['source_receiver_distance'].mean())
    cond_means = np.array(data.groupby('source_receiver_distance')['diff_pick_ot_time'].mean())
    plt.scatter(x = srd, y = cond_means,
            color = 'cornflowerblue',
            alpha = 0.3, marker = 'o',
            label = "conditional means")

    #Let's add a line of best fit using OLS regression
    linmod = sm.OLS(endog = y_train, exog = x_train)
    linmod_results = linmod.fit()
    pred = linmod_results.predict(x_train)

    #add the line to the plot
    ols, = plt.plot(x_train['source_receiver_distance'],
                pred, color = "red", linestyle = "dashed",
                label = "best fit line")

    #Let's also add some information about the estimated conditional median.
    median_model = smf.quantreg('diff_pick_ot_time ~ source_receiver_distance', data)
    median_model_results = median_model.fit(q = 0.5)
    pred_median = median_model_results.predict(x_train)
    median, = plt.plot(x_train['source_receiver_distance'],
                   pred_median, color = 'green',
                   linestyle = "dashed",
                   label = "median model fit")

    plt.legend()
    plt.show()
    
def multi_quantile_fitlines():
    """ 
    Plots a scatterplot of the source receiver distance vs the diff pick ot time.
    The plot includes the best fit lines of the 10th, 25th, 50th, 75th, ad 90th percentiles,
    calculated using quantile regression.
    """
    fig = plt.figure()
    fig.suptitle('DPOT vs SRD')
    plt.xlabel('Source Receiver Distance')
    plt.ylabel('DPOT')
    plt.scatter(x = data['source_receiver_distance'],
            y = data['diff_pick_ot_time'])

    coeff = []
    colors = ['orange', 'lime', 'gold', 'cyan', 'violet']
    i = 0
    handles = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    for q in quantiles:
        #build a model
        quant_model = smf.quantreg('diff_pick_ot_time ~ source_receiver_distance', data)
        #fit the model
        quant_mod_results = quant_model.fit(q = q)
        coeff.append(quant_mod_results.params['source_receiver_distance'])
        #get the estimates from the model
        quant_preds = quant_mod_results.predict(x_train)
        #plot the estimated values
        quantile, = plt.plot(x_train['source_receiver_distance'],
                         quant_preds, color = colors[i],
                         linestyle = "dashed",
                         label = str(int(q*100))+'th percentile Model')
        i = i + 1
        handles.append(quantile)

    plt.legend(handles=handles)
    plt.show()

if __name__ == "__main__":
    #run conditional_means_scatter
    conditional_means_scatter()
    #run multi_quantile_fitlines
    multi_quantile_fitlines()
    
    
    