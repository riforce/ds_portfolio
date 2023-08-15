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

from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#read in our training data
train_file = "~/repos/seismology_data/woodanderson_train.csv"
data = pd.read_csv(train_file)

y_train, x_train = dmatrices('diff_pick_ot_time ~ source_receiver_distance', data, return_type = 'dataframe')
x = data['source_receiver_distance']
y = data['diff_pick_ot_time']

x1 = pd.DataFrame({'source_receiver_distance':np.linspace(data.source_receiver_distance.min(), data.source_receiver_distance.max())})

poly_1st_order = smf.ols('diff_pick_ot_time ~ source_receiver_distance', data).fit()
poly_2nd_order = smf.ols('diff_pick_ot_time ~ source_receiver_distance + I(source_receiver_distance ** 2)', data).fit()
poly_3rd_order = smf.ols('diff_pick_ot_time ~ source_receiver_distance + I(source_receiver_distance ** 3)', data).fit()

prstd, iv_l, iv_u = wls_prediction_std(poly_2nd_order)
st, dat, ss2 = summary_table(poly_2nd_order, alpha = 0.05)

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
    
def poly_fitting(): 
    """ 
    Plots a scatterplot of the source receiver distance vs the diff pick ot time.
    The plot includes the polynomial fit lines of first, second, and third order,
    as well as the resultant models' R-squared values, calculated using OLS regression.
    """
    
    plt.figure(figsize = (8,6))
    plt.plot(x, y, 'o', alpha = 0.2)
    plt.title("Signal Velocity - Polynomial Fit")
    plt.plot(x1.source_receiver_distance,
             poly_1st_order.predict(x1),
             'r-',
             label = '1st order poly fit, $R^2$=%.2f' % poly_1st_order.rsquared)
    plt.plot(x1.source_receiver_distance,
             poly_2nd_order.predict(x1),
             'b-',
             label = '2nd order poly fit, $R^2$=%.2f' % poly_2nd_order.rsquared)
    plt.plot(x1.source_receiver_distance,
             poly_3rd_order.predict(x1),
             'g-',
             label = '3rd order poly fit, $R^2$=%.2f' % poly_3rd_order.rsquared)
    
    plt.legend(loc = "upper center", fontsize = 14)
    plt.show()
    
def second_order_confidence():
    """ 
    Plots a scatterplot of the source receiver distance vs the diff pick ot time.
    The plot includes second-order polynomial fit line calculated using OLS regression,
    as well as the 95% prediction and confidence intervals..
    """
    
    plt.figure(figsize=(8,6))
    plt.plot(x, y, 'o', alpha = 0.2, label = "")
    plt.title("Signal Velocity - Confidence Intervals")
    plt.plot(x1.source_receiver_distance, poly_2nd_order.predict(x1), 'k-', label = "2nd order polynomial fit")

    fittedvalues = dat[:,2]
    predict_mse = dat[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = dat[:,4:6].T
    predict_ci_low, predict_ci_upp = dat[:,6:8].T

    data_intervals = {'source_receiver_distance': x,
                      'srd_low': predict_ci_low,
                      'srd_upp': predict_ci_upp,
                      'conf_low': predict_mean_ci_low,
                      'conf_high': predict_mean_ci_upp}
    df_intervals = pd.DataFrame(data = data_intervals)
    df_intervals_sort = df_intervals.sort_values(by = "source_receiver_distance")
    
    plt.plot(df_intervals_sort.source_receiver_distance,
             df_intervals_sort.srd_low,
             color='r', linestyle='dashed',
             linewidth=2, label='95% prediction interval')
    
    plt.plot(df_intervals_sort.source_receiver_distance,
             df_intervals_sort.srd_upp,
             color='r', linestyle='dashed',
             linewidth=2, label='')

    plt.plot(df_intervals_sort.source_receiver_distance,
             df_intervals_sort.conf_low,
             color='c', linestyle='dashed',
             linewidth=2, label='95% confidence interval')

    plt.plot(df_intervals_sort.source_receiver_distance,
             df_intervals_sort.conf_high,
             color='c', linestyle='dashed',
             linewidth=2, label='')

    plt.legend(fontsize = 14)
    plt.show()    


def polynomial_quantile_regression():
    """ 
    Plots a scatterplot of the source receiver distance vs the diff pick ot time.
    The plot includes second-order polynomial fit line calculated using OLS regression,
    as well as the 5th, 25th, 50th, 75th, and 95th quantiles. 
    """
    #Now that we've visualized the confidence intervals around the 2nd-order polynomial fit, 
    #we can extend this quantile regression.
    mod = smf.quantreg(formula='diff_pick_ot_time ~ source_receiver_distance + I(source_receiver_distance ** 2)', data=data)
    
    # Quantile regression for 5 quantiles
    quantiles = [.05, .25, .50, .75, .95]
    
    # get all results in a list
    res_all = [mod.fit(q=q) for q in quantiles]
    
    #fit the model
    res_ols = smf.ols('diff_pick_ot_time ~ source_receiver_distance + I(source_receiver_distance ** 2)', data).fit()
    
    plt.figure(figsize=(8,6))
    
    # create x for prediction
    x_p = np.linspace(data['source_receiver_distance'].min(), data['source_receiver_distance'].max(), 50)
    df_p = pd.DataFrame({'source_receiver_distance': x_p})
    
    print("Generating curves...")
    for qm, res in zip(quantiles, res_all):
        # get prediction for the model and plot
        # here we use a dict which works the same way as the df in ols
        plt.plot(x_p, res.predict({'source_receiver_distance': x_p}), linestyle='--', lw=1, color='k')

        #apply the model
        y_ols_source_receiver_distance = res_ols.predict(df_p['source_receiver_distance'])
        #add model results to figure
        plt.plot(x_p, y_ols_source_receiver_distance, color='red', label='OLS')
        plt.scatter(data['source_receiver_distance'], data['diff_pick_ot_time'], alpha=.2)
        plt.xlabel("Distance (km)")
        plt.ylabel("Time (s)")
        plt.xlim((0, 250))
        plt.ylim((-10, 80))
        plt.title('QUANTILE REGRESSION', fontsize=14)
        
    plt.show()
        
if __name__ == "__main__":
    #run conditional_means_scatter
    conditional_means_scatter()
    #run multi_quantile_fitlines
    multi_quantile_fitlines()
    #run poly_fitting
    poly_fitting()
    #run second_order_confidence
    second_order_confidence()
    #run polynomial_quantile_regression
    polynomial_quantile_regression()
    
    
    