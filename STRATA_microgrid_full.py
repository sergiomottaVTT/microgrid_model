# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:49:23 2023

@author: sdsergio

Microgrid simulation model.
This model will implement load shifting behaviour and integration of BESS and EVs in a local microgrid.

Initial version developed between August - December 2023.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
# We need to create "Deep copies" of our arrays at some point in the code, hence why we import this module

import STRATA_functions as fn
import STRATA_mg_functions as mg
import mg_classes as cl

# %% DATA IMPORT
# Importing generation and load data, setting up the parameters of microgrid components
load_data = np.loadtxt(r'data/House1_LOADDATA_5900kWh-per-a.txt') #in kWh
gen_data = np.loadtxt(r'data/PV_pu_data.txt') #in pu
price_data = np.loadtxt(r'data/spotprice_oneyear.txt', usecols=2)

# How many days the simulation should run for
number_days = 1
# Setting the "sampling frequency" of the simulation: 15 minute intervals or 60 minute/hourly
minute_intervals = 60

# Load shifting is implemented
load_shifting = True
# Price to be considered as "expensive" and thus worth it to discharge BESS/EV/load shift
price_threshold = 0.05

# Setting up the parameters of the microgrid components
houses = 4
load_data = load_data * houses

### TO-DO: separate the loads!

# PV system
PV_installed_capacity = 50.0 #kWp
gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_parameters = {'capacity': 10, #capacity in kWh
                   'cRate': 1.5, #charge/discharge rate in kW
                   'SoC': 0.0, # initial SoC
                   'Control': 'price_threshold', #the type of control for the battery, will discharge when threshold is above setpoint
                   'Control setpoint': price_threshold,
                   'Grid enabled': False,
                   'SoC threshold': 1
                   }

# EVs
EV_parameters = {'number': 1,   # How many EVs connected to the microgrid
                 'capacity': 20, #capacity
                 'SoC': 20,     # initial SoC
                 'cRate': 0.5,   #charging rate of the charging station
                 'V2G': True,   #enabling V2G
                 'discharge threshold': (0.85, 0.6),    #can only be discharged if SoC > 85% capacity, down to 60% of capacity
                 }

# Modifying the data to be at the right size and frequency
load_data, gen_data, price_data, final_time, time_range = fn.modify_data(load_data, gen_data, price_data, number_days, minute_intervals)
# Creating an array of lists to keep track of when the loads will be shifted
#load_shift = [[] for _ in range(len(load_data))]
# %% Estimating the flexibility for each timestamp


# Setting load flexibility behaviour
#flexibility_curve, flexibility_window, pairs, minload, maxload, maxload_future = fn.define_flexibility(number_days, minute_intervals, load_data, plot=False)


# %% Monte Carlo: Generate multiple load & generation curves

def exponential_moving_average(data, alpha):
    smoothed_data = [data[0]]
    for _ in range(1, len(data)):
        smoothed_data.append(alpha*data[_] + (1-alpha)*smoothed_data[-1])
    return smoothed_data


def generate_random_curves(data, number_of_curves, variance, plotting=True):
    
    # We want to create N load curves with time intervals. So we will create 100 values for 00:00, 100 values for 01:00, and so forth.
    # initialising the array that will save our data
    random_curves = np.zeros([len(load_data), number_of_curves])

    # Creating random curves with NORMAL DISTRIBUTION
    for time in range(len(data)):
        
        N = number_of_curves #how many curves we want to create
        AVG = data[time] #the mean of the normal distribution
        STD = variance*AVG #the standard deviation of the normal distribution
        
        # Adding normally distributed random values for each time
        random_curves[time, :] = np.random.normal(loc=AVG, scale=STD, size=N)

    ### TO-DO: Curves need to be clipped for maximum and minimum values

    
    #### curve smoothing with exponential moving average
    alpha = 0.5
    smoothed_curves = np.ones_like(random_curves)
    for curve in range(random_curves.shape[1]):
        # running through all the N curves generated
        smoothed_curves[:, curve] = exponential_moving_average(random_curves[:, curve], alpha)


    if plotting==True:
        # Plotting the load curves we generated
        
        plt.figure()
        plt.title('Stochastic curves with normal distribution')
        plt.plot(random_curves, linestyle='--', linewidth=0.2)
        plt.plot(data, linestyle='-', linewidth=0.5, color='b', label='Original data curve')
        plt.legend()
        
      
        plt.figure()
        plt.title('Stochastic curves with normal distribution - SMOOTHED')
        plt.plot(smoothed_curves, linestyle='--', linewidth=0.2)
        plt.plot(np.mean(smoothed_curves, axis=1), color='k', linewidth=0.5, label='calculated average from random curves')
        plt.plot(data, linestyle='-', linewidth=0.5, color='b', label='Original data curve')
        plt.legend()

    return random_curves, smoothed_curves


#rdm_gen, smooth_gen = generate_random_curves(gen_data, 1000, 0.1)











####### 
# After the load is shifted with self-consumption and spot-price following, how much load is still available to be shifted?

# Industry load profile after the residential load profile














# %% Implementing some object-oriented programming for the first time in Python!

# Load objects are created with deep copies of the load array. Deep copies allow us to copy the full array, i.e. create a copy also of the objects
# within the array. A "shallow copy" copies only the array and references to objects contained within the original array!
load1 = cl.Load(load_data, load_data, True)
load2 = cl.Load(load_data*0.5, load_data*0.5, True)
load3 = cl.Load(load_data*0.3, load_data*0.3, True)

# Setting the load flexibility behaviour
load1.define_flexibility(number_days, minute_intervals, plot=False)
load2.define_flexibility(number_days, minute_intervals, plot=False)
load3.define_flexibility(number_days, minute_intervals, plot=False)

# Creating a list of all our loads
load_list = [load1, load2, load3]
# The total load of the microgrid and the total load considering also BESS and EVs as negative loads
total_demand = np.sum([load.load for load in load_list], axis=0)

total_demand_after_shift = np.sum([load.newload for load in load_list], axis=0)


# %% Transforming the EVs into objects

# Creating EVs and Setting EV behaviour
EV1 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], True)
EV1.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False)

EV2 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], True)
EV2.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=True)


EV_list = [EV1, EV2]


#%% MICROGRID OPERATION



# Flexibility availability calculation
# BESS_up, BESS_down, BESS_shift = [],[],[]
# gen_up, gen_down = [], []
# EV_up, EV_down, EV_shift = [],[],[]
# load_up, load_down = [], []

### TO-DO: Power-based tariffs!
### TO-DO: Monte Carlo simulation
### TO-DO: Calculate available flexibility at each timestamp
### TO-DO: Preliminary load shifts; then real-time load shifts

peak_limit = np.max(load1.newload)

# Performing the day-ahead load shifts
# "day-ahead" meaning that we perform load shifts from past and future timestamps, with a full-picture, as 
# opposed to being able to shift only future loads (and not past loads)

load_list, total_demand_after_shift = mg.mg_day_ahead_shifting(gen_data, load_list, total_demand_after_shift, price_data, peak_limit)
  
# And we now have the load shifts incorporated into the load objects and in the total demand
# We can now perform the microgrid operations for the day-ahead

BESS_SoC, BESS_io, EV_list, grid_io = mg.mg_day_ahead_operation(BESS_parameters, EV_list, gen_data, total_demand_after_shift, price_data, 
                                                                price_threshold, minute_intervals)

# %%  Assigning values to a dataframe for easier inspection


# Checking if the microgrid is operating OK
total_EV_io = np.sum([ev.EV_io for ev in EV_list], axis=0)

checksum = total_demand_after_shift - gen_data + BESS_io + total_EV_io + grid_io

if np.sum(checksum) != 0:
    print('Warning! Something strange in the microgrid, energy is leaking somewhere...')
else:
    print("Microgrid operating as expected")
# Assigning loads to the dataframe
  

# Creating the data from other parameters
microgrid_data = {
    'Total demand': total_demand,
    'Total demand_shift': total_demand_after_shift,
    'Generation': gen_data,
    'BESS_SoC': BESS_SoC,
    'BESS charge/discharge': BESS_io,
    'Grid import/export': grid_io,
    'Price data': price_data,
    #'Upreg': load_up,
    #'Downreg': load_down
    }
# Creating the multiple load data
for i, load in enumerate(load_list, start=1):
    microgrid_data[f'Load {i}'] = load.load
    microgrid_data[f'Shifted load {i}'] = load.newload
    microgrid_data[f'Load shift {i}'] = load.load_shift
# Creating the multiple EV data
for i, ev in enumerate(EV_list, start=1):
    microgrid_data[f'EV{i} I/O'] = ev.EV_io
    microgrid_data[f'EV{i} SoC'] = ev.EV_SoC
    microgrid_data[f'EV{i} plugged'] = ev.plugged_array


microgrid_simulation = pd.DataFrame(microgrid_data, index=time_range)



# %% Evaluating results

#savings, benefit, self_sufficiency, self_consumption = fn.result_eval(microgrid_simulation, minute_intervals)

#microgrid_simulation.plot()

# %%

# plt.figure()
# plt.plot(newload)
# plt.plot(load_up)
# plt.plot(load_down)

# # %% 

# plt.figure()
# plt.plot(load_data)
# plt.plot(grid_io)



