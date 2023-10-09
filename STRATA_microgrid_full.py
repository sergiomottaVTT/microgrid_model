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
number_days = 2
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
                   'Control': 'load_threshold', #the type of control for the battery, will discharge when threshold is above setpoint
                   'Control setpoint': 3.0
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
flexibility_curve, flexibility_window, pairs, minload, maxload = fn.define_flexibility(number_days, minute_intervals, load_data)

#%% MICROGRID OPERATION

# initialising simulation values
load_shift = np.zeros_like(load_data)
grid_io, BESS_SoC, BESS_io, EV_SoC, EV_io = [], [], [], [], []
newload = load_data.copy()



for timestamp in range(len(newload)):
    # I: check excess generation
    difference = gen_data[timestamp] - newload[timestamp]
    # II: Setting the BESS behaviour
    difference, BESS_parameters, BESS_SoC, BESS_io = mg.BESS_behaviour(difference, BESS_parameters, BESS_SoC, BESS_io, price_data, timestamp)
    # III: Setting the EV behaviour
    
    # If we have excess generation, the EV will will always try to charge    
    difference, EV_parameters, EV_SoC, EV_io, EV_out_for_the_day = mg.EV_behaviour(timestamp, difference, EV_plugged, EV_parameters, 
                                                                                   price_data, price_threshold, battery_usage_percentages, 
                                                                                   minute_intervals, EV_out_for_the_day, EV_SoC, EV_io)

    # IV: Setting the load shifting behaviour
    if load_shifting == True:
        difference, newload = mg.load_shift_behaviour(difference, newload, timestamp, pairs, load_shift, flexibility_curve, price_data)   
    # V: Setting the grid import/export behaviour
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
    'EVPlug': EV_plugged
    },
    index=time_range)


# %% Evaluating results

savings, benefit, self_sufficiency, self_consumption = fn.result_eval(microgrid_simulation, minute_intervals)

microgrid_simulation.plot()




