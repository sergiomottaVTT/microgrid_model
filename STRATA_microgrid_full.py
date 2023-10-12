# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:49:23 2023

@author: sdsergio

Building the whole microgrid operation
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import STRATA_functions as fn
import STRATA_mg_functions as mg

# %% DATA IMPORT
# Importing generation and load data, setting up the parameters of microgrid components
load_data = np.loadtxt(r'data/House1_LOADDATA_5900kWh-per-a.txt') #in kWh
gen_data = np.loadtxt(r'data/PV_pu_data.txt') #in pu
price_data = np.loadtxt(r'data/spotprice_oneyear.txt', usecols=2)

# How many days the simulation should run for
number_days = 1
# Setting the "sampling frequency" of the simulation: 15 minute intervals or 60 minute/hourly
minute_intervals = 15

# Load shifting is implemented
load_shifting = True
# Price to be considered as "expensive" and thus worth it to discharge BESS/EV/load shift
price_threshold = 0.05

# Setting up the parameters of the microgrid components
houses = 4
load_data = load_data * houses
# PV system
PV_installed_capacity = 50.0 #kWp
gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_parameters = {'capacity': 10, #capacity in kWh
                   'cRate': 1.5, #charge/discharge rate in kW
                   'SoC': 0.0, # initial SoC
                   'Control': 'price_threshold', #the type of control for the battery, will discharge when threshold is above setpoint
                   'Control setpoint': price_threshold,
                   'Grid enabled': True,
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
# Initial flag variable for the EV simulation
EV_out_for_the_day = 0

# Modifying the data to be at the right size and frequency
load_data, gen_data, price_data, final_time, time_range = fn.modify_data(load_data, gen_data, price_data, number_days, minute_intervals)

# %% Estimating the flexibility for each timestamp

# Setting EV behaviour
battery_usage_percentages, EV_plugged = fn.set_EV_behaviour(EV_parameters, number_days, final_time, minute_intervals, load_data, plot=False)

# Setting load flexibility behaviour
flexibility_curve, flexibility_window, pairs, minload, maxload, maxload_future = fn.define_flexibility(number_days, minute_intervals, load_data, plot=True)




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










#%% MICROGRID OPERATION

# initialising simulation values
load_shift = np.zeros_like(load_data)
grid_io, BESS_SoC, BESS_io, EV_SoC, EV_io = [], [], [], [], []
newload = load_data.copy()

# Flexibility availability calculation
BESS_up, BESS_down, BESS_shift = [],[],[]
gen_up, gen_down = [], []
EV_up, EV_down, EV_shift = [],[],[]
load_up, load_down = [], []



for timestamp in range(len(newload)):
    # I: check excess generation
    difference = gen_data[timestamp] - newload[timestamp]
    # II: Setting the BESS behaviour
    difference, BESS_parameters, BESS_SoC, BESS_io = mg.BESS_behaviour(difference, BESS_parameters, BESS_SoC, BESS_io, price_data, timestamp)
    # III: Setting the EV behaviour   
    difference, EV_parameters, EV_SoC, EV_io, EV_out_for_the_day = mg.EV_behaviour(timestamp, difference, EV_plugged, EV_parameters, 
                                                                                   price_data, price_threshold, battery_usage_percentages, 
                                                                                   minute_intervals, EV_out_for_the_day, EV_SoC, EV_io)

    # IV: Setting the load shifting behaviour
    if load_shifting == True:
        difference, newload, load_shift = mg.load_shift_behaviour(difference, newload, load_data, 
                                                                  timestamp, pairs, load_shift, flexibility_curve, price_data)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # V: Calculating the available flexibility in the microgrid at each timestamp
    
    # BESS can up/down regulate by charging/discharging
    BESS_up.append(min(BESS_parameters['SoC'], BESS_parameters['cRate']))
    BESS_down.append(min(BESS_parameters['capacity'] - BESS_parameters['SoC'], BESS_parameters['cRate']))
    # The BESS can also up/down regulate by just shifting the time which it charge/discharge
    BESS_shift.append(BESS_io)
    
    
    # Generation can only up-regulate by injecting power to the grid, and down-regulating by not injecting power to the grid (curtailing)
    if difference > 0:
        gen_up.append(difference)
        gen_down.append(difference)
    else:
        gen_up.append(0)
        gen_down.append(0)
    
    
    # EV can up/down regulate by charging/discharging (if V2G is enabled) - and only if it is plugged!
    if EV_plugged[timestamp] == 1:
        if (EV_parameters['V2G'] == True) and (EV_parameters['SoC'] > EV_parameters['discharge threshold'][0]*EV_parameters['capacity']):
            EV_up.append(min(EV_parameters['cRate'], (EV_parameters['SoC'] - EV_parameters['discharge threshold'][1]*EV_parameters['capacity'])))
        else:
            EV_up.append(0)
        
        EV_down.append(min(EV_parameters['capacity'] - EV_parameters['SoC'], EV_parameters['cRate']))
        EV_shift.append(EV_io)
    else:
        EV_up.append(0); EV_down.append(0); EV_shift.append(0)
    
    
    # Load flexibility
    
    # up-regulation: reducing the load
    # Was the load already shifted?
    if load_shift[timestamp] < 0:
        load_up.append(0)
    else:
        load_up.append(newload[timestamp] - flexibility_curve[timestamp]*newload[timestamp])
    
    
    # down-regulation: increasing the load
    # which timestamps the load can be moved FROM?
    # Calculating the timestamp range (from flexibility curve) from which timestamps could be shifted
    idx_shiftable_loads = [pair[0] for pair in pairs if timestamp in pair[1]]
    # However, we can't alter the past, so we can only get indexes which are larget than timestamp
    idx_shiftable_loads = [element for element in idx_shiftable_loads if element > timestamp]
    
    maxload_timestamp = newload[timestamp]
    for idx in idx_shiftable_loads:
        if load_shift[idx] >= 0:
            maxload_timestamp = maxload_timestamp + flexibility_curve[timestamp]*newload[idx]
    
    load_down.append(maxload_timestamp)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # VI: Setting the grid import/export behaviour
    grid_import, grid_export, grid_io = mg.grid_behaviour(difference, grid_io)
    
    
# Assigning values to a dataframe for easier inspection

microgrid_simulation = pd.DataFrame({
    #'Hour': range(1, len(load_data)+1),
    'Original Load': load_data,
    'New Load': newload,
    'Generation': gen_data,
    'BESS_SoC': BESS_SoC,
    'BESS charge/discharge': BESS_io,
    'EV_SoC': EV_SoC,
    'EV charge/discharge': EV_io,
    'Grid import/export': grid_io,
    'Price data': price_data,
    'EVPlug': EV_plugged,
    'Load shift': load_shift,
    'Upreg': load_up,
    'Downreg': load_down
    },
    index=time_range)


# %% Evaluating results

savings, benefit, self_sufficiency, self_consumption = fn.result_eval(microgrid_simulation, minute_intervals)

microgrid_simulation.plot()

# %%

plt.figure()
plt.plot(newload)
plt.plot(load_up)
plt.plot(load_down)

# %% 

plt.figure()
plt.plot(load_data)
plt.plot(grid_io)



