<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:34:03 2023

@author: sdsergio

Initial version of a stochastic simulation model for a microgrid

"""

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
# for the gamma distribution function
import scipy.stats as stats

# Importing generation and load data, setting up the parameters of microgrid components

load_data = np.loadtxt(r'data/House1_LOADDATA_5900kWh-per-a.txt') #in kWh
gen_data = np.loadtxt(r'data/PV_pu_data.txt') #in pu

#%% Setting up the parameters of the microgrid components

# PV system
PV_installed_capacity = 50.0 #kWp
gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_capacity = 10 #kWh
BESS_cRate = 2.5 #kW charge/discharge
BESS_initial_SoC = 0.0 #initial state-of-charge


#%% Performing a simulation

### TODO: make this a function

# We want to test just for 3 days

load_data = load_data[:168]
gen_data = gen_data[:168]

# initialising simulation values

grid_import_export= []
BESS_SoC = []
BESS_io = []

BESS_current_SoC = BESS_initial_SoC

for hour in range(len(load_data)):
    # calculating the excess generation available from the local generation at this hour
    excess_gen = gen_data[hour] - load_data[hour]
    
    # RULE I: maximising self-consumption
    # as we want to maximise self-consumption, the BESS will charge/discharge as often as possible

    if excess_gen >= 0:    # if there is excess generation, the BESS will be charged
        BESS_charge = min(excess_gen, BESS_cRate, BESS_capacity - BESS_current_SoC)
        BESS_current_SoC = min(BESS_current_SoC + BESS_charge, BESS_capacity)
        BESS_discharge = 0
        
        # if we have excess generation, we'd only export to the grid
        grid_export = excess_gen - BESS_charge
        grid_import = 0
        
    else:        # this means the generation is not enough to meet all the load, and we want to discharge
        BESS_discharge = min(BESS_cRate, abs(excess_gen), BESS_current_SoC)
        BESS_current_SoC = max(BESS_current_SoC - BESS_discharge, 0)
        BESS_charge = 0
        
        # since we can't meet the load with the generation, we import from the grid what is needed
        grid_import = min((excess_gen + BESS_discharge), 0) #excess_gen is <0, BESS_discharge is >0, import should be negative
        grid_export = 0
    
    # combining grid import and export (+ is export, - is import)
    grid_import_export.append(grid_import + grid_export)
    
    BESS_SoC.append(BESS_current_SoC)
    BESS_io.append(BESS_charge - BESS_discharge)


# Assigning values to a dataframe for easier inspection

microgrid_simulation = pd.DataFrame({
    #'Hour': range(1, len(load_data)+1),
    'Load': load_data,
    'Generatio': gen_data,
    'Grid import/export': grid_import_export,
    'BESS_SoC': BESS_SoC,
    'BESS charge/discharge': BESS_io
    })

# Evaluating the results    
microgrid_simulation.plot()



# %% When adding stochasticity to the load, generation, and grid availability, we must be mindful that these are variable also with the time of the year
# Thus, we will have the average samples for 365 days (8760 values) and we'll generate N random curves following a normal distribution with the samples
# for each day serving as the average values. This will lengthen our Monte Carlo simulation, but will serve as a more accurate representation of the
# expected values. We can utilise pandas' datetime to ensure we always get the appropriate dates.


#%% Adding stochasticity in the load behaviour

# Normal distribution of the loads

plt.figure()
plt.plot(load_data)

# The "mold" for a daily consumption profile is the curve below
dr_load = load_data[25:49]
plt.figure()
plt.plot(dr_load)
plt.title('Average daily load profile')


# However, this load profile should have some stochasticity - there's variance in when the peaks occur, and how much is each peak, etc.
# This variance is studied in references, so we should get better numbers here

time_variance = 1.0     # a time variance of 1 hours up or down **THIS IS NOT YET IMPLEMENTED
load_variance = 0.15    # a variance of 15% of the load value give or take

# This means that 25% of the load values could be shifted up-or-down 1 hour

# The hourly load values should have a normal distribution with the average being the value of the "mold"
# and the standard deviation being 25% of the average


# We want to create N load curves with time intervals. So we will create 100 values for 00:00, 100 values for 01:00, and so forth.
number_of_curves = 1000
# initialising the array that will save our data
load_curves = np.zeros([len(dr_load), number_of_curves])

# Creating random load curves and adding stochasticity to it
for time in range(len(dr_load)):
    
    N = number_of_curves #how many curves we want to create
    AVG = dr_load[time] #the mean of the normal distribution
    STD = load_variance*AVG #the standard deviation of the normal distribution
    
    # Adding normally distributed random values for each time
    load_curves[time, :] = np.random.normal(loc=AVG, scale=STD, size=N)
    
# Plotting the load curves we generated

plt.figure()
plt.title('Stochastic load curves with normal distribution')
plt.plot(load_curves, linestyle='--', linewidth=0.2)
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()

# The random load curves have too much of a "sharp" difference between subsequent values. This can be seen from this plot:
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load,  label='average load curve')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# Which is not fully in line with the reality - typically the loads tend to be smoother, as in the average load curve.

# To mitigate this effect from the stochasticity, 
# I asked ChatGPT for a solution and it suggested me to implement a moving average or exponential moving average smoothing.

# Trying a moving average
window_size = 2
smoothed_array = np.convolve(load_curves[:, 1], np.ones(window_size)/window_size, mode='valid')

# Testing to see if it worked
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load, label='average load curve')
plt.plot(smoothed_array, linestyle='--', label='smoothed array')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# It worked well, but we need to add the padding as the smoothed array is only 22 values.

# Now trying the exponential moving average to see

def exponential_moving_average(data, alpha):
    smoothed_data = [data[0]]
    for _ in range(1, len(data)):
        smoothed_data.append(alpha*data[_] + (1-alpha)*smoothed_data[-1])
    return smoothed_data

alpha = 0.6
smoothed_array_exponential = exponential_moving_average(load_curves[:,1], alpha)

# Testing to see if it worked
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load, label='average load curve')
plt.plot(smoothed_array, linestyle='--', label='smoothed array MA')
plt.plot(smoothed_array_exponential, linestyle='--', label='smoothed array EMA')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# Pretty impressive results! The EMA seems to be more reliable in keeping the original behaviour of the random curve with the peaks keeping an alpha of 0.5
# However, we must keep in mind that smoothing the curves will result in a skewed average load value from the generated load curves.
# Higher alpha (i.e. lower smoothing) brings the average closer to the expected.

# Now if we apply the exponential moving average to all the generated load curves
load_curves_smoothed = np.ones_like(load_curves)

for curve in range(load_curves.shape[1]):
    # running through all the N curves generated
    smoothed_curve = exponential_moving_average(load_curves[:, curve], alpha)
    load_curves_smoothed[:, curve] = smoothed_curve
    

# And comparing the original generated load curves with the smoothed ones
plt.figure()
plt.title('Stochastic load curves with normal distribution')
plt.plot(load_curves, linestyle='--', linewidth=0.2)
plt.plot(np.mean(load_curves, axis=1), color='k', linewidth=0.5, label='calculated average from random curves')
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()

plt.figure()
plt.title('Stochastic load curves with normal distribution - SMOOTHED')
plt.plot(load_curves_smoothed, linestyle='--', linewidth=0.2)
plt.plot(np.mean(load_curves_smoothed, axis=1), color='k', linewidth=0.5, label='calculated average from random curves')
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()


# %% Creating supporting functions to generate load samples

def create_load_intervals(load, time, magnitude, **kwargs):
    """
    Function to create the value intervals for flexible loads. It describes the average load values over a duration of time, the minimum possible values,
    and the maximum possible values according to the flexibility available in the load.
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)
    
    time:       INT
                What is the time interval to which the load can be shifted, i.e., the window of +/- hours which accept the flexibility of the load.
        
    magnitude:  FLOAT
                How much load can be shifted, in decimal format. This will be the value removed from the original load and added to the loads in the time
                interval.
                
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    
    Returns:
    ---------
    pairs:      LIST OF TUPLES
                List of tuples containing the indexes to which the load can be shifted for each hour. For example, load at h=4 can be shifted to h=2, 3, 5, 6
                if the time = 2. In such case, the total window of flexibility is 4 hours, centered in (but not including) h=4.
    
    minload:    ARRAY
                Array of the same shape as _load_ with the minimum possible value of the loads accounting for the load shift to other times.
                
    maxload:    ARRAY
                Array of the same shape as _load_ with the maximum possible value of the loads accounting for the load shift to other times.
            
    """
    # INITIALISING THE ARRAYS WE USE FOR CALCULATING MINIMUM AND MAXIMUM LOAD VALUES
    maxload, minload, addload = np.zeros_like(load), np.zeros_like(load), np.zeros_like(load)
    indexes = np.array(range(len(load)))
    shift = indexes.copy()
    pairs = []
    
    plot = kwargs.get('plot')
    
    # CALCULATING THE LOAD VALUES
    # the minimum load in each time is the original load - the amount of load that can be shifted
    minload = load - magnitude * load
    
    
    # the maximum load is more complex to calculate, as multiple hours may shift to one time
       
    for hour in range(len(load)):
        # we iterate over each hourly value of the load

        # how much load will be added to previous and subsequent hours
        load_shift = magnitude * load[hour]
        
        # Gettng the indices in the load vector to be changed:
        # we create a rolling array
        shift = np.roll(shift, -1)
        idx_neg = hour - time  # if the index is negative, it already rolls over for the previous day
        
        # creating a list without the central index (since we want to change the indices -2, -1, 1, 2 and NOT the index 0, for example)
        prev = list(range(idx_neg, hour))
        post = list(shift[0:time])     # this is the "cat's jump" here, the rolling list from 0 to time_variance results the indices that we want
        symmetrical_indexes = prev + post
        
        # just to save the hours and the indexes
        pairs.append((hour, symmetrical_indexes))
        
        for indices in symmetrical_indexes:
            addload[indices] = addload[indices] + load_shift

    # the maximum load possible is the combination of the "normal" load and the potential for all load shifting from nearby hours
    maxload = load + addload

    # Plotting
    if plot == True:
        plt.figure()
        plt.plot(minload, linestyle='--', alpha=0.7, label='min load with flexibility')
        plt.plot(maxload, linestyle='--', alpha=0.7, label='max load with flexibility')
        plt.plot(load, color='k', label='average load')
        plt.legend()
        plt.title('Load values considering the load shifting potential')
        plt.xlabel('Time (h)')
        plt.ylabel('Load (kWh)')


    return pairs, minload, maxload
### TODO: Make this function allow for different load distributions (right now it's only Gamma)

pairs, minload, maxload = create_load_intervals(dr_load, 2, 0.15, plot=True)

def generate_load_probability_curves(load, time, flexibility_probability, magnitude, pairs, distribution_type, **kwargs):
    """
    Function to generate random load curves that follow an appropriate behaviour to represent the distribution given.
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)
    
    time:       INT
                What is the time interval to which the load can be shifted, i.e., the window of +/- hours which accept the flexibility of the load.
   
    flexibility_probability: FLOAT
                What is the probability that a load shift actually occurs     
   
    magnitude:  FLOAT
                How much load can be shifted, in decimal format. This will be the value removed from the original load and added to the loads in the time
                interval.
                
    pairs:      LIST OF TUPLES
                List of tuples to show what are the intervals to which the loads can be shifted
                
                
    distribution_type:  STRING
                String defining which distribution type should be followed for the generation of the random load curves
                
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    
    Returns:
    ---------
    hour_load_values:ARRAY
                3D Array of possible load values for each hour at the shape of (HOUR, load values)
                
    hour_load_probabilities: ARRAY
                3D Array with the possible load probabilities (for each value and each hour)
    
            
    """
    
    if distribution_type == "Gamma":
        
        # The probability that the load shift goes to each adjacent time interval
        prob_adjacent_interval = flexibility_probability/(time*2)
        
        hour_load_values = []
        hour_load_probabilities = []
        # for each hour
        for i in range(len(load)):
            
            # initialising variables
            load_values, prob_values, all_combinations = [], [], []
            
            # we create the total number of load shift combinations
            for r in range(1, len(pairs[hour][1])+1):
                combinations_r = itertools.combinations(pairs[hour][1], r)
                all_combinations.extend(combinations_r)

            # all the possible combinations of adjacent loads being shifted to this hour are created and are used to find the load 
            #intervals and probabilities

            # if there is no load reduction at that hour
            for combination in all_combinations:

                load_added = load[i]
                
                for element in combination:
                    load_added = load_added + magnitude * load[element]
                
                load_values.append(load_added)    
                probability = prob_adjacent_interval ** len(combination)
                prob_values.append(probability)

            # if there is load reduction at that hour
            for combination in all_combinations:

                load_added_reduction = load[i] - magnitude * load[i]
                
                for element in combination:
                    load_added_reduction = load_added_reduction + magnitude * load[element]
                
                load_values.append(load_added_reduction)    
                probability = flexibility_probability * prob_adjacent_interval ** len(combination)
                prob_values.append(probability)

            # Appending the probability of reduction
            load_values.append(load[i] - magnitude * load[i])
            prob_values.append(flexibility_probability)
            
            # Appending the probability of nothing happening and the value being the average (1 - all other probabilities)
            load_values.append(load[i])
            prob_values.append((1 - np.sum(np.array(prob_values))))

            # Appending for each hour
            hour_load_values.append(load_values)
            hour_load_probabilities.append(prob_values)


    return hour_load_values, hour_load_probabilities
### TODO: Implement normal distribution
### TODO: Implement variable probability per timeslot

hour_load_values, hour_load_probabilities = generate_load_probability_curves(dr_load, 2, 0.15, 0.15, pairs, "Gamma")

# %% 
def generate_random_load_curves(load, hour_load_values, hour_load_probabilities, distribution_type, N, **kwargs):
    """
    Function to generate N samples of load curves following the probability distribution specified by the inputs
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)

    hour_load_values:ARRAY
                3D Array of possible load values for each hour at the shape of (HOUR, load values)
                
    hour_load_probabilities: ARRAY
                3D Array with the possible load probabilities (for each value and each hour)              
                
    distribution_type:  STRING
                String defining which distribution type should be followed for the generation of the random load curves
    
    N:          INT
                How many random samples to generate
    
    kwargs:
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    sample_analysis: STRING
                String indicating the level of analysis to be performed on the generated samples. "Full" gives a plot and text, "Text" gives only text analysis
    minload, maxload: ARRAY
                Arrays with the minimum and maximum value the loads can take, calculated by create_load_intervals function, and used on the plot
    
    Outputs:
    ---------
    random_samples:     ARRAY
                        Array of [len(load) x N] with N random samples in the shape of _load_ that represent the load with the given distribution type.
    
    """

    plot = kwargs.get('plot')
    sample_analysis = kwargs.get('sample_analysis')
    minload = kwargs.get('minload')
    maxload = kwargs.get('maxload')
    
    if distribution_type == 'Gamma':
        # For a gamma distribution, we need to calculate the shape (k) and scale (theta) parameters 
        
        hourly_samples= []
        for hour in range(len(load)):
            # unpacking
            load_values = np.array(hour_load_values[hour])
            prob_values = np.array(hour_load_probabilities[hour])
            
            # Calculating the parameters for the Gamma distribution
            mean = np.sum(load_values * prob_values)/1  #1 because the probabilities (weights) add up to 1
            variance = np.sum(prob_values * (load_values - mean) ** 2)/1

            k = (mean**2)/variance
            theta = variance/mean
            
            # Generating the PDF
            x = np.linspace(min(load_values), max(load_values), 1000)
            pdf_estimated = stats.gamma.pdf(x, a=k, scale=theta)
            
            # Generating random values           
            samples = []
            while len(samples) != N:
                random_generated_values = stats.gamma.rvs(a=k, scale=theta, size=N - len(samples))
                #plt.hist(random_generated_values, density=True, bins='auto', histtype='stepfilled')
                samples.extend(random_generated_values[(min(load_values) <= random_generated_values) & (random_generated_values <= max(load_values))])

            if plot == True:
                plt.figure()
                plt.hist(samples, density=True, bins='auto', histtype='stepfilled')
                plt.title('Distribution of load values with a ' + str(N) + ' sample size for hour = ' + str(hour))
                plt.plot(x, pdf_estimated, label='Estimated PDF')
                plt.legend()
                plt.xlabel('Load value (kWh)')
        
        
            hourly_samples.append(samples)
            
        # converting a list to an array for ease of use
        hourly_samples = np.array(hourly_samples)
        
        
        if sample_analysis != None:
            random_samples_load_consumption = []
            for i in hourly_samples.T:
                random_samples_load_consumption.append(np.sum(i))
        
        if sample_analysis == 'Full':
            plt.figure()
            for i in hourly_samples.T:
                plt.plot(i, alpha=0.3, linewidth=0.3)
            
            plt.plot(load, linewidth=1, color='k')
            plt.plot(minload, linestyle='--', color='b')
            plt.plot(maxload, linestyle='--', color='g')
            print('Total load consumption of the original load sample is {:.2f}'.format(np.sum(load))) 
            print('Average total load consumption for the random samples is {:.2f}'.format(np.mean(random_samples_load_consumption)))
        
        elif sample_analysis == 'Text':
            print('Total load consumption of the original load sample is {:.2f}'.format(np.sum(load)))
            print('Average total load consumption for the random samples is {:.2f}'.format(np.mean(random_samples_load_consumption))) 
    
    return hourly_samples

random_samples = generate_random_load_curves(dr_load, hour_load_values, hour_load_probabilities, 'Gamma', 100000, 
                                             plot=False, minload=minload, maxload=maxload, sample_analysis='Text')


### TODO: Why the random samples have more load than the original load?
### FIXME: Correct this issue!



#%%


















=======
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:34:03 2023

@author: sdsergio

Initial version of a stochastic simulation model for a microgrid

"""

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
# for the gamma distribution function
import scipy.stats as stats

# Importing generation and load data, setting up the parameters of microgrid components

load_data = np.loadtxt(r'data/House1_LOADDATA_5900kWh-per-a.txt') #in kWh
gen_data = np.loadtxt(r'data/PV_pu_data.txt') #in pu

#%% Setting up the parameters of the microgrid components

# PV system
PV_installed_capacity = 50.0 #kWp
gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_capacity = 10 #kWh
BESS_cRate = 2.5 #kW charge/discharge
BESS_initial_SoC = 0.0 #initial state-of-charge


#%% Performing a simulation

### TODO: make this a function

# We want to test just for 3 days

load_data = load_data[:168]
gen_data = gen_data[:168]

# initialising simulation values

grid_import_export= []
BESS_SoC = []
BESS_io = []

BESS_current_SoC = BESS_initial_SoC

for hour in range(len(load_data)):
    # calculating the excess generation available from the local generation at this hour
    excess_gen = gen_data[hour] - load_data[hour]
    
    # RULE I: maximising self-consumption
    # as we want to maximise self-consumption, the BESS will charge/discharge as often as possible

    if excess_gen >= 0:    # if there is excess generation, the BESS will be charged
        BESS_charge = min(excess_gen, BESS_cRate, BESS_capacity - BESS_current_SoC)
        BESS_current_SoC = min(BESS_current_SoC + BESS_charge, BESS_capacity)
        BESS_discharge = 0
        
        # if we have excess generation, we'd only export to the grid
        grid_export = excess_gen - BESS_charge
        grid_import = 0
        
    else:        # this means the generation is not enough to meet all the load, and we want to discharge
        BESS_discharge = min(BESS_cRate, abs(excess_gen), BESS_current_SoC)
        BESS_current_SoC = max(BESS_current_SoC - BESS_discharge, 0)
        BESS_charge = 0
        
        # since we can't meet the load with the generation, we import from the grid what is needed
        grid_import = min((excess_gen + BESS_discharge), 0) #excess_gen is <0, BESS_discharge is >0, import should be negative
        grid_export = 0
    
    # combining grid import and export (+ is export, - is import)
    grid_import_export.append(grid_import + grid_export)
    
    BESS_SoC.append(BESS_current_SoC)
    BESS_io.append(BESS_charge - BESS_discharge)


# Assigning values to a dataframe for easier inspection

microgrid_simulation = pd.DataFrame({
    #'Hour': range(1, len(load_data)+1),
    'Load': load_data,
    'Generatio': gen_data,
    'Grid import/export': grid_import_export,
    'BESS_SoC': BESS_SoC,
    'BESS charge/discharge': BESS_io
    })

# Evaluating the results    
microgrid_simulation.plot()



# %% When adding stochasticity to the load, generation, and grid availability, we must be mindful that these are variable also with the time of the year
# Thus, we will have the average samples for 365 days (8760 values) and we'll generate N random curves following a normal distribution with the samples
# for each day serving as the average values. This will lengthen our Monte Carlo simulation, but will serve as a more accurate representation of the
# expected values. We can utilise pandas' datetime to ensure we always get the appropriate dates.


#%% Adding stochasticity in the load behaviour

# Normal distribution of the loads

plt.figure()
plt.plot(load_data)

# The "mold" for a daily consumption profile is the curve below
dr_load = load_data[25:49]
plt.figure()
plt.plot(dr_load)
plt.title('Average daily load profile')


# However, this load profile should have some stochasticity - there's variance in when the peaks occur, and how much is each peak, etc.
# This variance is studied in references, so we should get better numbers here

time_variance = 1.0     # a time variance of 1 hours up or down **THIS IS NOT YET IMPLEMENTED
load_variance = 0.15    # a variance of 15% of the load value give or take

# This means that 25% of the load values could be shifted up-or-down 1 hour

# The hourly load values should have a normal distribution with the average being the value of the "mold"
# and the standard deviation being 25% of the average


# We want to create N load curves with time intervals. So we will create 100 values for 00:00, 100 values for 01:00, and so forth.
number_of_curves = 1000
# initialising the array that will save our data
load_curves = np.zeros([len(dr_load), number_of_curves])

# Creating random load curves and adding stochasticity to it
for time in range(len(dr_load)):
    
    N = number_of_curves #how many curves we want to create
    AVG = dr_load[time] #the mean of the normal distribution
    STD = load_variance*AVG #the standard deviation of the normal distribution
    
    # Adding normally distributed random values for each time
    load_curves[time, :] = np.random.normal(loc=AVG, scale=STD, size=N)
    
# Plotting the load curves we generated

plt.figure()
plt.title('Stochastic load curves with normal distribution')
plt.plot(load_curves, linestyle='--', linewidth=0.2)
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()

# The random load curves have too much of a "sharp" difference between subsequent values. This can be seen from this plot:
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load,  label='average load curve')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# Which is not fully in line with the reality - typically the loads tend to be smoother, as in the average load curve.

# To mitigate this effect from the stochasticity, 
# I asked ChatGPT for a solution and it suggested me to implement a moving average or exponential moving average smoothing.

# Trying a moving average
window_size = 2
smoothed_array = np.convolve(load_curves[:, 1], np.ones(window_size)/window_size, mode='valid')

# Testing to see if it worked
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load, label='average load curve')
plt.plot(smoothed_array, linestyle='--', label='smoothed array')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# It worked well, but we need to add the padding as the smoothed array is only 22 values.

# Now trying the exponential moving average to see

def exponential_moving_average(data, alpha):
    smoothed_data = [data[0]]
    for _ in range(1, len(data)):
        smoothed_data.append(alpha*data[_] + (1-alpha)*smoothed_data[-1])
    return smoothed_data

alpha = 0.6
smoothed_array_exponential = exponential_moving_average(load_curves[:,1], alpha)

# Testing to see if it worked
plt.figure()
plt.plot(load_curves[:,1], label='random load curve')
plt.plot(dr_load, label='average load curve')
plt.plot(smoothed_array, linestyle='--', label='smoothed array MA')
plt.plot(smoothed_array_exponential, linestyle='--', label='smoothed array EMA')
plt.legend()
plt.title('Sharp variation between susequent values in a random load curve')

# Pretty impressive results! The EMA seems to be more reliable in keeping the original behaviour of the random curve with the peaks keeping an alpha of 0.5
# However, we must keep in mind that smoothing the curves will result in a skewed average load value from the generated load curves.
# Higher alpha (i.e. lower smoothing) brings the average closer to the expected.

# Now if we apply the exponential moving average to all the generated load curves
load_curves_smoothed = np.ones_like(load_curves)

for curve in range(load_curves.shape[1]):
    # running through all the N curves generated
    smoothed_curve = exponential_moving_average(load_curves[:, curve], alpha)
    load_curves_smoothed[:, curve] = smoothed_curve
    

# And comparing the original generated load curves with the smoothed ones
plt.figure()
plt.title('Stochastic load curves with normal distribution')
plt.plot(load_curves, linestyle='--', linewidth=0.2)
plt.plot(np.mean(load_curves, axis=1), color='k', linewidth=0.5, label='calculated average from random curves')
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()

plt.figure()
plt.title('Stochastic load curves with normal distribution - SMOOTHED')
plt.plot(load_curves_smoothed, linestyle='--', linewidth=0.2)
plt.plot(np.mean(load_curves_smoothed, axis=1), color='k', linewidth=0.5, label='calculated average from random curves')
plt.plot(dr_load, linestyle='-', linewidth=0.5, color='b', label='Average daily load')
plt.legend()


# %% Creating supporting functions to generate load samples

def create_load_intervals(load, time, magnitude, **kwargs):
    """
    Function to create the value intervals for flexible loads. It describes the average load values over a duration of time, the minimum possible values,
    and the maximum possible values according to the flexibility available in the load.
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)
    
    time:       INT
                What is the time interval to which the load can be shifted, i.e., the window of +/- hours which accept the flexibility of the load.
        
    magnitude:  FLOAT
                How much load can be shifted, in decimal format. This will be the value removed from the original load and added to the loads in the time
                interval.
                
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    
    Returns:
    ---------
    pairs:      LIST OF TUPLES
                List of tuples containing the indexes to which the load can be shifted for each hour. For example, load at h=4 can be shifted to h=2, 3, 5, 6
                if the time = 2. In such case, the total window of flexibility is 4 hours, centered in (but not including) h=4.
    
    minload:    ARRAY
                Array of the same shape as _load_ with the minimum possible value of the loads accounting for the load shift to other times.
                
    maxload:    ARRAY
                Array of the same shape as _load_ with the maximum possible value of the loads accounting for the load shift to other times.
            
    """
    # INITIALISING THE ARRAYS WE USE FOR CALCULATING MINIMUM AND MAXIMUM LOAD VALUES
    maxload, minload, addload = np.zeros_like(load), np.zeros_like(load), np.zeros_like(load)
    indexes = np.array(range(len(load)))
    shift = indexes.copy()
    pairs = []
    
    plot = kwargs.get('plot')
    
    # CALCULATING THE LOAD VALUES
    # the minimum load in each time is the original load - the amount of load that can be shifted
    minload = load - magnitude * load
    
    
    # the maximum load is more complex to calculate, as multiple hours may shift to one time
       
    for hour in range(len(load)):
        # we iterate over each hourly value of the load

        # how much load will be added to previous and subsequent hours
        load_shift = magnitude * load[hour]
        
        # Gettng the indices in the load vector to be changed:
        # we create a rolling array
        shift = np.roll(shift, -1)
        idx_neg = hour - time  # if the index is negative, it already rolls over for the previous day
        
        # creating a list without the central index (since we want to change the indices -2, -1, 1, 2 and NOT the index 0, for example)
        prev = list(range(idx_neg, hour))
        post = list(shift[0:time])     # this is the "cat's jump" here, the rolling list from 0 to time_variance results the indices that we want
        symmetrical_indexes = prev + post
        
        # just to save the hours and the indexes
        pairs.append((hour, symmetrical_indexes))
        
        for indices in symmetrical_indexes:
            addload[indices] = addload[indices] + load_shift

    # the maximum load possible is the combination of the "normal" load and the potential for all load shifting from nearby hours
    maxload = load + addload

    # Plotting
    if plot == True:
        plt.figure()
        plt.plot(minload, linestyle='--', alpha=0.7, label='min load with flexibility')
        plt.plot(maxload, linestyle='--', alpha=0.7, label='max load with flexibility')
        plt.plot(load, color='k', label='average load')
        plt.legend()
        plt.title('Load values considering the load shifting potential')
        plt.xlabel('Time (h)')
        plt.ylabel('Load (kWh)')


    return pairs, minload, maxload
### TODO: Make this function allow for different load distributions (right now it's only Gamma)

pairs, minload, maxload = create_load_intervals(dr_load, 2, 0.15, plot=True)

def generate_load_probability_curves(load, time, flexibility_probability, magnitude, pairs, distribution_type, **kwargs):
    """
    Function to generate random load curves that follow an appropriate behaviour to represent the distribution given.
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)
    
    time:       INT
                What is the time interval to which the load can be shifted, i.e., the window of +/- hours which accept the flexibility of the load.
   
    flexibility_probability: FLOAT
                What is the probability that a load shift actually occurs     
   
    magnitude:  FLOAT
                How much load can be shifted, in decimal format. This will be the value removed from the original load and added to the loads in the time
                interval.
                
    pairs:      LIST OF TUPLES
                List of tuples to show what are the intervals to which the loads can be shifted
                
                
    distribution_type:  STRING
                String defining which distribution type should be followed for the generation of the random load curves
                
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    
    Returns:
    ---------
    hour_load_values:ARRAY
                3D Array of possible load values for each hour at the shape of (HOUR, load values)
                
    hour_load_probabilities: ARRAY
                3D Array with the possible load probabilities (for each value and each hour)
    
            
    """
    
    if distribution_type == "Gamma":
        
        # The probability that the load shift goes to each adjacent time interval
        prob_adjacent_interval = flexibility_probability/(time*2)
        
        hour_load_values = []
        hour_load_probabilities = []
        # for each hour
        for i in range(len(load)):
            
            # initialising variables
            load_values, prob_values, all_combinations = [], [], []
            
            # we create the total number of load shift combinations
            for r in range(1, len(pairs[hour][1])+1):
                combinations_r = itertools.combinations(pairs[hour][1], r)
                all_combinations.extend(combinations_r)

            # all the possible combinations of adjacent loads being shifted to this hour are created and are used to find the load 
            #intervals and probabilities

            # if there is no load reduction at that hour
            for combination in all_combinations:

                load_added = load[i]
                
                for element in combination:
                    load_added = load_added + magnitude * load[element]
                
                load_values.append(load_added)    
                probability = prob_adjacent_interval ** len(combination)
                prob_values.append(probability)

            # if there is load reduction at that hour
            for combination in all_combinations:

                load_added_reduction = load[i] - magnitude * load[i]
                
                for element in combination:
                    load_added_reduction = load_added_reduction + magnitude * load[element]
                
                load_values.append(load_added_reduction)    
                probability = flexibility_probability * prob_adjacent_interval ** len(combination)
                prob_values.append(probability)

            # Appending the probability of reduction
            load_values.append(load[i] - magnitude * load[i])
            prob_values.append(flexibility_probability)
            
            # Appending the probability of nothing happening and the value being the average (1 - all other probabilities)
            load_values.append(load[i])
            prob_values.append((1 - np.sum(np.array(prob_values))))

            # Appending for each hour
            hour_load_values.append(load_values)
            hour_load_probabilities.append(prob_values)


    return hour_load_values, hour_load_probabilities
### TODO: Implement normal distribution
### TODO: Implement variable probability per timeslot

hour_load_values, hour_load_probabilities = generate_load_probability_curves(dr_load, 2, 0.15, 0.15, pairs, "Gamma")

# %% 
def generate_random_load_curves(load, hour_load_values, hour_load_probabilities, distribution_type, N, **kwargs):
    """
    Function to generate N samples of load curves following the probability distribution specified by the inputs
    
    Parameters:
    ----------
    load:       ARRAY
                Original load values (typically average hourly values over a 24-hour cycle)

    hour_load_values:ARRAY
                3D Array of possible load values for each hour at the shape of (HOUR, load values)
                
    hour_load_probabilities: ARRAY
                3D Array with the possible load probabilities (for each value and each hour)              
                
    distribution_type:  STRING
                String defining which distribution type should be followed for the generation of the random load curves
    
    N:          INT
                How many random samples to generate
    
    kwargs:
    plot:       BOOL
                True or False value to define if there should be a plot output showing the min, max and average loads.
    sample_analysis: STRING
                String indicating the level of analysis to be performed on the generated samples. "Full" gives a plot and text, "Text" gives only text analysis
    minload, maxload: ARRAY
                Arrays with the minimum and maximum value the loads can take, calculated by create_load_intervals function, and used on the plot
    
    Outputs:
    ---------
    random_samples:     ARRAY
                        Array of [len(load) x N] with N random samples in the shape of _load_ that represent the load with the given distribution type.
    
    """

    plot = kwargs.get('plot')
    sample_analysis = kwargs.get('sample_analysis')
    minload = kwargs.get('minload')
    maxload = kwargs.get('maxload')
    
    if distribution_type == 'Gamma':
        # For a gamma distribution, we need to calculate the shape (k) and scale (theta) parameters 
        
        hourly_samples= []
        for hour in range(len(load)):
            # unpacking
            load_values = np.array(hour_load_values[hour])
            prob_values = np.array(hour_load_probabilities[hour])
            
            # Calculating the parameters for the Gamma distribution
            mean = np.sum(load_values * prob_values)/1  #1 because the probabilities (weights) add up to 1
            variance = np.sum(prob_values * (load_values - mean) ** 2)/1

            k = (mean**2)/variance
            theta = variance/mean
            
            # Generating the PDF
            x = np.linspace(min(load_values), max(load_values), 1000)
            pdf_estimated = stats.gamma.pdf(x, a=k, scale=theta)
            
            # Generating random values           
            samples = []
            while len(samples) != N:
                random_generated_values = stats.gamma.rvs(a=k, scale=theta, size=N - len(samples))
                #plt.hist(random_generated_values, density=True, bins='auto', histtype='stepfilled')
                samples.extend(random_generated_values[(min(load_values) <= random_generated_values) & (random_generated_values <= max(load_values))])

            if plot == True:
                plt.figure()
                plt.hist(samples, density=True, bins='auto', histtype='stepfilled')
                plt.title('Distribution of load values with a ' + str(N) + ' sample size for hour = ' + str(hour))
                plt.plot(x, pdf_estimated, label='Estimated PDF')
                plt.legend()
                plt.xlabel('Load value (kWh)')
        
        
            hourly_samples.append(samples)
            
        # converting a list to an array for ease of use
        hourly_samples = np.array(hourly_samples)
        
        
        if sample_analysis != None:
            random_samples_load_consumption = []
            for i in hourly_samples.T:
                random_samples_load_consumption.append(np.sum(i))
        
        if sample_analysis == 'Full':
            plt.figure()
            for i in hourly_samples.T:
                plt.plot(i, alpha=0.3, linewidth=0.3)
            
            plt.plot(load, linewidth=1, color='k')
            plt.plot(minload, linestyle='--', color='b')
            plt.plot(maxload, linestyle='--', color='g')
            print('Total load consumption of the original load sample is {:.2f}'.format(np.sum(load))) 
            print('Average total load consumption for the random samples is {:.2f}'.format(np.mean(random_samples_load_consumption)))
        
        elif sample_analysis == 'Text':
            print('Total load consumption of the original load sample is {:.2f}'.format(np.sum(load)))
            print('Average total load consumption for the random samples is {:.2f}'.format(np.mean(random_samples_load_consumption))) 
    
    return hourly_samples

random_samples = generate_random_load_curves(dr_load, hour_load_values, hour_load_probabilities, 'Gamma', 100000, 
                                             plot=False, minload=minload, maxload=maxload, sample_analysis='Text')


### TODO: Why the random samples have more load than the original load?
### FIXME: Correct this issue!



#%%


















>>>>>>> origin/main
