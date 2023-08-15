# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:15:18 2023

@author: riley
"""

#useful imports
import numpy as np
import csv
from pandas import *
import pandas as pd
import h5py


#the test residuals from UNet
resid_file = "~/repos/seismology_data/test_residuals.csv"

#these are the expected values from the testing dataset
amplitude_file = "~/repos/seismology_data/woodanderson_test.csv"

#this is wavefrom data from the testing dataset
waveformData = h5py.File("C:/Users/riley/repos/seismology_data/mlWoodAnderson_test_waveforms.h5", "r")

residuals = pd.read_csv(resid_file) 
test = pd.read_csv(amplitude_file) 

lag = residuals['true_lag']
resids = residuals['residual']
true_amp = test['amplitude']*0.01 #converted to meters
estimate = lag - resids

#Peak-to-peak amplitude finder
def find_next_amplitude(t0 : float,
                        window_start_time : float,
                        dt : float,
                        signal,
                        min_max_signal,
                        original_dt : float,
                        tol_pct : float = 10,
                        min_change_pct : float = 5,
                        n_peaks : int = 5):
    """
    Attempts to find the next peak corresponding to this amplitude.

    Parameters
    ---------- d
    t0 : float
       The start time of the trace in UTC seconds since the epoch.
    window_start_time : float
       The start time of the search window in UTC seconds since the epoch.
       This should be the result of an ML model and indicates the first
       peak in the peak-to-peak amplitude calculation.
    dt : float
       The sampling period of the signal in seconds.
    signal : np.array
       The signal.  This should have been `deconvolved' to a Wood-Anderson
       instrument.
    min_max_signal : np.array
       This signal is -1 at a local signal minimum, +1 at a local signal
       maximum, and 0 otherwise.
    original_dt : float
       The original sampling period of the signal in second.
       Effectively what happens was the interpolation can move where the
       (min) or maximum was.
    tol_pct : float
       Basically, we let small high-frequency signals swing back resulting
       in a false-peak.  If that small-peak's amplitude is less than this
       percentage then it is ignored and the algorithm keeps searching for
       the next peak.
    min_change_pct : float
       Sometimes, there's some high frequency garbage near the peak which
       results in a false min or max.  If that peak-to-peak amplitude
       is less than min_change_pct*signal(window_start_time) then that
       peak is ignored and we keep searching for the next peak.
    n_peaks : int
       The number of additional peaks to scrutinize.

    Returns
    -------
    j0 : int
       The index in the signal corresponding to the first peak.
    j1 : int
       The index in the signal corresponding to the second peak.
    window_start_time : float
       The refined start time of the signal provided the signal was
       resampled.
    """
    start_index = int( (window_start_time - t0)/dt + 0.5 )

    # May have to do some searching
    if (original_dt != dt): #If the original sampling period is not equal to the sampling period of the signal...
        start_index0 = start_index

        # Upsampling was done by sinc interpolation.
        # The original pick was on the `lower-resolution' signal.
        # Graphically, the problem looks as follows.  Call the x's the
        # original samples and the -'s the upsampled points.
        # x    x    x
        # - - - - - -
        # In this case, the pick would be at the center x but the min/max
        # must now exist on the dashed grid.

        if (dt < original_dt): #If the sampling period of the signal is less than the original sampling period
            n_window = int(round(original_dt/dt)) #rounded difference between original_dt and dt
            i0 = start_index - n_window
            i1 = start_index + n_window + 1 # Exclusive
        else: #otherwise, the sampling period of the signal is greater than or equal to the original sampling period.
            # Going from fine to coarse (downsampling)
            n_window = 1
            i0 = start_index - n_window
            i1 = start_index + n_window + 1 # Exclusive

        n_min_max = np.count_nonzero(min_max_signal[i0:i1]) #Gets all the values of 1 and -1 (maximums and minimums) between i0 and i1.

        if (n_min_max < 1): #There were no values of -1 or 1 in the window.
            print("Could not find min/max in window",
                  min_max_signal[i0:i1])
            return None, None, window_start_time

        if (n_min_max > 1): #There were multiple values of -1 and/or 1 in the window.
            print("I don't know what to do here")
            return None, None, window_start_time

        #This for-loop shifts the window over to start at a min/max.
        for i in range(i0,i1): #Looping through the window looking for min/max
            if (min_max_signal[i] != 0): #If we find one...
                window_start_time = t0 + i*dt #set the start of the window plus i number of sampling periods.
                start_index = i #set the start index to where the min/max was found.
                break
        print("Original and new start indices:", start_index0, start_index)

    if (start_index >= len(min_max_signal)):
        print("Exceeded end of min/max signal - skipping")
        return None, None, window_start_time

    #If the start of the window is not at a min/max, we'll try to fix the pick.
    if (min_max_signal[start_index] == 0):
        for i in range(-1,2):
            if (min_max_signal[start_index + i] != 0): #If there is some next index where we find a min/max...
                start_index = start_index + i #We'll set the start index equal to that index.
                break
        window_start_time = t0 + start_index*dt #and from the start index we'll get the new start of the window.
        print("Fixed pick by a sample")
        if (min_max_signal[start_index] == 0): #if that STILL didn't work, we know the pick didn't start at a min/max.
            print("Pick does not start at min/max",
                  min_max_signal[start_index-1:start_index+2])
            return None, None, window_start_time

    # Now that we are starting at a maximum let's do the next one
    j0 = start_index #The index in the signal corresponding to the first peak is the start index.
    n_samples = len(signal)

    # Idea is to scrutinize the next n_peaks local mins/maxs
    # and find a peak-to-peak ratio that exceeds the minimum
    win_amps = []
    for i in range(n_peaks):
        # Scan until the end of the signal...
        for j in range(j0 + 1, n_samples):
            # New local min/max -> check it out
            if (min_max_signal[j] != 0):
                j1 = j
                amplitude = signal[j0] - signal[j1]
                d_change = abs(amplitude)/abs(signal[j0])*100.
                # If percent change is tiny then keep trucking
                if (d_change < min_change_pct):
                    continue
                #print(j0, j1, amplitude/signal[j0]*100.)
                d = {'start_index' : j0,
                     'end_index' : j1,
                     'amplitude_sign' : int(np.sign(amplitude)),
                     'abs_amplitude' : abs(amplitude)}
                win_amps.append(d)
                j0 = j1 # Update
                #print(d)
                break
        # Look
        n_wins = len(win_amps)
        if (n_wins == 1):
            continue
        # Does this window exceed the turning ratio
        sign0 = win_amps[0]['amplitude_sign']
        sign1 = win_amps[n_wins - 1]['amplitude_sign']
        amp0 = win_amps[0]['abs_amplitude']
        amp1 = win_amps[n_wins - 1]['abs_amplitude']
        # Is this amp-window going the other way?
        if (sign0 != sign1):
            # Have we gotten above the ratio?
            ratio = amp1/amp0*100
            #print("Ratio for window:", n_wins, ratio)
            if (ratio > tol_pct):
                #print("Recommend:", sign0, sign1, start_index*dt, dt*win_amps[n_wins - 1]['start_index'])
                return start_index, win_amps[n_wins - 1]['start_index'], window_start_time
        # End check
    # Loop on windows

    # That didn't work.  The goal really is to just return some result now
    j0 = start_index
    windows = []
    for k in range(3):
        # Scan until end
        for j in range(j0 + 1, n_samples):
            if (min_max_signal[j] != 0):
                j1 = j
                amplitude = signal[j0] - signal[j1]
                windows.append( [j0, j1, np.sign(amplitude), abs(amplitude)] )
                j0 = j1
                break
    # Is the amplitude in the second window less than the tolerance?
    if (len(windows) < 1):
        print("Algorithm failure - failed to end window")
        return None, None, window_start_time
    j0 = windows[0][0]
    j1 = windows[0][1]
    if (len(windows) < 3):
        print("Could not find another window", j0)
        return windows[0][0], windows[0][1], window_start_time
    # Direction change in next window
    if (windows[0][2] != windows[1][2]):
        ratio = windows[1][3]/windows[0][3]*100
        #print("Ratio:", ratio)
        if (ratio < tol_pct):
            j1 = windows[2][1]
            #windows[2][1], window_start_time
    else:
        print("Shouldn't have to min/min or max/max in a row")
        return None, window_start_time
    assert j1 > j0, 'this is backwards'
    return j0, j1, window_start_time



#We need to integrate the waveform information from the h5 file with the residual information and the test observations.

def make_row(i, waveformData, estimate, true_amp):
    """
    Parameters
    -----------
    i : numeric
       An index.
    waveformData : File
       The waveform data h5 file.
    estimate : series
       Prediction residuals subtracted from the true-lag. In other words, UNet's estimates of the amplitude.
    true_amp : series
        Test-set amplitudes.

    Returns
    --------
    row : list
         A list representing a row of a data frame, including an index corresponding to the row number, the index in the signal corresponding to the first peak, the index in the signal corresponding to the second peak, the refined start time of the signal provided the signal was resampled, the estimated index of the signal's peak, the predicted amplitude, the lag of the observation, the true amplitude of the observation, and the difference between the true and predicted amplitude.
    """
    #extract the signal from the waveform data
    signal = waveformData["X"][i,:,0]

    #get if it is a max or a min
    min_max_signal = waveformData["X"][i,:,1]

    #get the estimated amplitude
    est_i = estimate[i]

    #check if the estimate index is a max or a min
    for j in range(est_i, 0, -1):
      if min_max_signal[j] != 0:
        est_i = j
        break

    #try using the find_next_amplitude function to find a peak-to-peak amplitude.
    #If successful, construct a row from the index, start and end of the signal,
    #the window start time, the index of the estimate, the amplitude, the lag, the
    #true amplitude, and the difference between the amplitude and the true amplitude
    dt = 0.01
    try:
      j0, j1, window_start_time = find_next_amplitude(t0 = 0,
                                                        window_start_time = est_i*dt,
                                                        dt = dt,
                                                        signal = signal,
                                                        min_max_signal = min_max_signal,
                                                        original_dt = dt,
                                                        tol_pct = 10,
                                                        min_change_pct = 5,
                                                        n_peaks = 5)
      amp = abs(signal[j1]-signal[j0])
      diff = true_amp[i] - amp
      row = [i, j0, j1, window_start_time, est_i, amp, lag[i], true_amp[i], diff]

    #if the find_next_amplitude function cannot find a peak_to_peak amplituede,
    #construct a row out of the index, index of the estimate, lag, and true amplitude.
    except:
      row = [i, None, None, None, est_i, None, lag[i], true_amp[i], None]

    return row

#we'll write the combined waveform + test set + residuals data into a new csv.

new_f = open("C:/Users/riley/repos/seismology_data/amplitude_residuals.csv", "w")
clean_f = open("C:/Users/riley/repos/seismology_data/clean_amplitude_residuals.csv", "w")
writer1 = csv.writer(new_f)
writer2 = csv.writer(clean_f)
s = waveformData["X"].shape[0]

header = ["index","pick_start","pick_end","window_start_time", "est_i", "amplitude", "lag_at_index", "observed_amp", "difference"]
writer1.writerow(header)
writer2.writerow(header)

for row in range(s):
  new_data = make_row(row, waveformData, estimate, true_amp)
  writer1.writerow(new_data)
  if new_data[5] is not None:
    writer2.writerow(new_data)

new_f.close()
clean_f.close()

