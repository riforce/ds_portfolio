# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:15:14 2023

@author: riley
"""

#useful imports
import numpy as np
import pandas as pd
from pandas import *
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
import statistics

#read in data
train_file = "~/repos/seismology_data/woodanderson_train.csv"
data = pd.read_csv(train_file)


#Visualizing wave velocities

def srd_v_dpot():
    """
    Plots a scatterplot of srd vs dpot, along with a best-fit line.

    Returns
    -------
    slope : numpy.ndarray
        The slope and intercept of the best fit line

    """
    fig, ax = plt.subplots()
    ax.plot(data['source_receiver_distance'], data['diff_pick_ot_time'],
            'o', color = "cornflowerblue", alpha = 0.2)

    plt.title("Source Receiver Distance vs DPOT")
    plt.xlabel('source receiver distance')
    plt.ylabel('diff pick ot time')
    
    #adding a line of best fit to the scatterplot
    slope = np.polyfit(data['source_receiver_distance'],
                       data['diff_pick_ot_time'], 1)
    p = np.poly1d(slope)
    plt.plot(data['source_receiver_distance'],
             p(data['source_receiver_distance']),
             color='darkblue')

    plt.show()
    return slope


def srd_v_dpot_with_baselines():
    """
    Plots the scatterplot of srd vs dpot, with additional reference velocities:
        0.5 km/s (a basin S wave velocity)
        2.9 km/s (typical crustal S wave velocity)
        4.3 km/s (typical mantle S wave velocity)
        
    Returns
    -------
    None.

    """
    #Add known baseline velocities to the previous plot:
    fig, ax = plt.subplots()
    ax.plot(data['source_receiver_distance'], data['diff_pick_ot_time'],
            'o', color = "cornflowerblue", alpha = 0.2)
    plt.title("Source Receiver Distance vs DPOT")
    plt.xlabel('source receiver distance')
    plt.ylabel('diff pick ot time')
    
    a = np.linspace(0,250,100)
    basin = 0.5*a
    crustal = 2.9*a
    mantle = 4.3*a
    
    ax.plot(a, basin, '-r', label='basin S wave velocity')
    ax.plot(a, crustal, '-b', label='crustal S wave velocity')
    ax.plot(a, mantle, '-g', label='mantle S wave velocity')
    
    ax.legend()
    plt.show()


def source_velocity_event_type_comparison():
    """
    Plot comparing the signal velocity scatterplots of quarry blast and 
    earthquake events.

    Returns
    -------
    slope_qb : numpy.ndarray
        Slope and intercept of quarry blast fit line
    slope_le : numpy.ndarray
        Slope and intercept of earthquake fit line

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (15,5))
    size = 13
    
    #grabbing subsets of the training data
    qb_data = data[data['etype'] == 'qb']
    le_data = data[data['etype'] == 'le']
    
    #the plot on ax1 will be the quarry blast data
    ax1.plot(qb_data['source_receiver_distance'],
             qb_data['diff_pick_ot_time'],
             'o', color = "cornflowerblue", alpha = 0.2)
    ax1.set_title("SRD vs DPOT - Quarry Blasts", fontsize = size)
    ax1.set(xlabel = "source receiver distance",
            ylabel = "diff pick ot time")
    
    #finding a line of best fit and printing the slope & intercept
    slope_qb = np.polyfit(qb_data['source_receiver_distance'],
                          qb_data['diff_pick_ot_time'], 1)
    p_qb = np.poly1d(slope_qb)
    ax1.plot(qb_data['source_receiver_distance'],
             p_qb(qb_data['source_receiver_distance']),
             color = "black")
    
    #the plot on ax2 will be the earthquake data
    ax2.plot(le_data['source_receiver_distance'],
             le_data['diff_pick_ot_time'],
             'o', color = 'cornflowerblue', alpha = 0.2)
    ax2.set_title("SRD vs DPOT - Earthquakes", fontsize = size)
    ax2.set(xlabel = "source receiver distance",
            ylabel = "diff pick ot time")
    
    #finding a line of best fit and printing the slope & intercept
    slope_le = np.polyfit(le_data['source_receiver_distance'],
                          le_data['diff_pick_ot_time'], 1)
    p_le = np.poly1d(slope_le)
    ax2.plot(le_data['source_receiver_distance'],
             p_le(le_data['source_receiver_distance']), color = "black")
    
    plt.show()
    return slope_qb, slope_le


def magnitude_histogram(n_bins):
    """
    Plots a histogram of the magnitudes and returns useful statistics. 

    Parameters
    ----------
    n_bins : int
        number of bins for plotting

    Returns
    -------
    mean : float
        the mean magnitude
    std : float
        standard deviation of the magnitude
    q1 : float
        first quartile of the magnitude (25th percentile)
    q3 : float
        third quartile of the magnitude (75th percentile)

    """    
    mags = data["event_magnitude"]
    
    fig, ax = plt.subplots(figsize = (10, 7))
    N, bins, patches = ax.hist(mags, n_bins)
    ax.set_title("Histogram of Event Magnitudes")
    ax.set(xlabel = "magnitude", ylabel = "count")
    
    ax.grid(color = "darkgreen", alpha = 0.5)
    ax.xaxis.set_tick_params(pad = 10)
    ax.yaxis.set_tick_params(pad = 10)
    
    #adding gradient color
    fractions = ((N**(1/20))/N.max())
    norm = colors.Normalize(fractions.min(), fractions.max())
    
    for frac, patch in zip(fractions, patches):
        color = plt.cm.viridis(norm(frac))
        patch.set_facecolor(color)
        
    mean = statistics.mean(mags)
    std = statistics.stdev(mags)
    q1, q3 = np.percentile(mags, [25, 75])
        
    plt.show()
    return mean, std, q1, q3


def amplitude_histogram():
    """
    Plots a histogram of the amplitudes and returns useful statistics.

    Returns
    -------
    mean : float
        the mean amplitude
    std : float
        the standard deviation of the amplitude
    q1 : float
        the first quartile of the amplitude (25th percentile)
    q3 : float
        the third quartile of the amplitude (75th percentile)

    """
    #Visualizing amplitudes
    
    amps = data['amplitude']
    bins_arange = np.arange(1,80,2)
    
    #constructing the figure
    fig, ax = plt.subplots(figsize = (10, 7), tight_layout = True)
    N, bins, patches = ax.hist(amps, bins_arange)
    ax.set_title("Histogram of Amplitude")
    ax.set(xlabel = "Amplitude (m)", ylabel = "count")
    
    ax.grid(color = "darkgreen", alpha = 0.5)
    ax.xaxis.set_tick_params(pad = 10)
    ax.yaxis.set_tick_params(pad = 10)
    
    #adding gradient color
    fractions = ((N**(1/20))/N.max())
    norm = colors.Normalize(fractions.min(), fractions.max())
    
    for frac, patch in zip(fractions, patches):
        color = plt.cm.viridis(norm(frac))
        patch.set_facecolor(color)
        
        #printing some useful statistics
    mean = statistics.mean(amps)
    std = statistics.stdev(amps)
    q1, q3 = np.percentile(amps, [25, 75])
        
    plt.show()
    return mean, std, q1, q3


#I'm interested in whether the amplitude distributions for quarry blasts and earthquakes are different:

def compare_event_type_amplitude_histogram():
    """
    Plots histograms of amplitudes for both quarry blasts and earthquakes, using
    the same scale to compare their distributions.

    Returns
    -------
    None.

    """
    #let's look at the distributions of qb and le amplitudes side by side.
    qb_amps = data[data["etype"] == "qb"]["amplitude"]
    le_amps = data[data["etype"] == "le"]["amplitude"]
    size = 25
    
    #sharing the y axis will show us the distributions on the same scale
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 8), sharey = True)
    bins = np.arange(1, 80, 2)
    
    ax1.hist(qb_amps, bins)
    ax1.set_title("QB Amplitudes", fontsize = size)
    ax1.grid(color = "cornflowerblue")
    
    ax2.hist(le_amps, bins)
    ax2.set_title("Earthquake Amplitudes", fontsize = size)
    ax2.grid(color = "cornflowerblue")

    plt.show()




if __name__ == "__main__":
    print("SIGNAL VELOCITY", "\n")
    #plot srd vs dpot
    srdvdpot_slope = srd_v_dpot()
    print("slope, intercept of srd vs dpot:", srdvdpot_slope)

    #plot srd vs dpot with additional baseline velocities
    srd_v_dpot_with_baselines()
    
    #compare the signal velocities of quarry blasts and earthquakes
    slope_qb, slope_le = source_velocity_event_type_comparison()
    print("quarry blast slope, intercept", slope_qb)
    print("earthquake slope, intercept", slope_le)
    print("\n")
    
    print("MAGNITUDES", "\n")
    #visualize magnitude with a gradient-colored histogram with 50 bins
    mag_mean, mag_std, mag_q1, mag_q3 = magnitude_histogram(50)
    print("Mean magnitude:", mag_mean)
    print("Standard deviation of magnitude:", mag_std)
    print("q1, q3 of magnitude:", mag_q1, ",", mag_q3)
    print("\n")
    
    print("AMPLITUDES", "\n")
    #visualize amplitude with a gradient-colored histogram
    amp_mean, amp_std, amp_q1, amp_q3 = amplitude_histogram()
    print("Mean amplitude:", amp_mean)
    print("Standard deviation of amplitude:", amp_std)
    print("q1, q3 of amplitude:", amp_q1, ",", amp_q3)
    
    #compare quarry blasts and earthquakes by amplitude distribution
    compare_event_type_amplitude_histogram()


