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

import copy
# We need to create "Deep copies" of our arrays at some point in the code, hence why we import this module

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
# Initial flag variable for the EV simulation
EV_out_for_the_day = 0

# Modifying the data to be at the right size and frequency
load_data, gen_data, price_data, final_time, time_range = fn.modify_data(load_data, gen_data, price_data, number_days, minute_intervals)

load_shift = [[] for _ in range(len(load_data))]
# %% Estimating the flexibility for each timestamp

# Setting EV behaviour
battery_usage_percentages, EV_plugged = fn.set_EV_behaviour(EV_parameters, number_days, final_time, minute_intervals, load_data, plot=False)

# Setting load flexibility behaviour
flexibility_curve, flexibility_window, pairs, minload, maxload, maxload_future = fn.define_flexibility(number_days, minute_intervals, load_data, plot=False)




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

class Load:
    def __init__(self, load, newload, load_shift, flexibility_curve, flexibility_interval, shifting):
        self.load = copy.deepcopy(load)
        self.newload = copy.deepcopy(newload)
        self.load_shift = copy.deepcopy(load_shift)
        self.flexibility_curve = copy.deepcopy(flexibility_curve)
        self.flexibility_interval = copy.deepcopy(flexibility_interval)
        self.shifting = shifting


# Load objects are created with deep copies of the load array. Deep copies allow us to copy the full array, i.e. create a copy also of the objects
# within the array. A "shallow copy" copies only the array and references to objects contained within the original array!
load1 = Load(load_data, load_data, load_shift, flexibility_curve, flexibility_window, True)
load2 = Load(load_data*0.5, load_data*0.5, load_shift, flexibility_curve, flexibility_window, True)
load3 = Load(load_data*0.3, load_data*0.3, load_shift, flexibility_curve, flexibility_window, True)


# Creating a list of all our loads
load_list = [load1, load2, load3]
# The total load of the microgrid and the total load considering also BESS and EVs as negative loads
total_demand = np.sum([load.load for load in load_list], axis=0)

total_demand_after_shift = np.sum([load.newload for load in load_list], axis=0)




# %% Transforming the EVs into objects

class EV:
    def __init__(self, capacity, cRate, SoC, discharge_threshold, EV_plugged, battery_use, V2G=False):
        self.capacity = capacity
        self.cRate = cRate
        self.SoC = copy.deepcopy(SoC)
        self.discharge_threshold = discharge_threshold
        self.plugged_array = copy.deepcopy(EV_plugged)
        self.battery_use = battery_use
        self.V2G = V2G
        self.day_disconnect = 0
        self.EV_SoC = []
        self.EV_io = []
        
# Setting EV behaviour
battery_usage_percentages, EV_plugged = fn.set_EV_behaviour(EV_parameters, number_days, final_time, minute_intervals, load_data, plot=False)

EV1 = EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'],
         EV_plugged, battery_usage_percentages, True)

# Setting EV behaviour
battery_usage_percentages_2, EV_plugged_2 = fn.set_EV_behaviour(EV_parameters, number_days, final_time, minute_intervals, load_data, plot=False, random=True)

EV2 = EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'],
         EV_plugged_2, battery_usage_percentages_2, True)

EV_list = [EV1, EV2]


#%% MICROGRID OPERATION

# initialising simulation values
grid_io, BESS_SoC, BESS_io, EV_SoC, EV_io = [], [], [], [], []

# Flexibility availability calculation
BESS_up, BESS_down, BESS_shift = [],[],[]
gen_up, gen_down = [], []
EV_up, EV_down, EV_shift = [],[],[]
load_up, load_down = [], []

### TO-DO: Power-based tariffs!
### TO-DO: Peark power as toggle flag
### TO-DO: Monte Carlo simulation
### TO-DO: Calculate available flexibility at each timestamp
### TO-DO: Preliminary load shifts; then real-time load shifts


spot_price_following = True
gen_shift = False
peak_limit = np.max(load1.newload)



# Performing the day-ahead load shifts

for timestamp in range(len(load_data)):
    
    
    ###### I: check excess generation  
    difference = gen_data[timestamp] - total_demand_after_shift[timestamp]
    
    ##### II: Load shifting
    
    if difference >= 0 and gen_shift == True:
        # we want to perform a load shift TOWARDS this time to increase the self-consumption
        
        for load in load_list:
            # Calculating the timestamp range (from flexibility curve) from which timestamps could be shifted
            idx_shiftable_loads = [pair[0] for pair in pairs if timestamp in pair[1]]
            
            # Since we are not doing the load shifting in real-time, but rather 
            # We do it time-based, not price-based. I.e. we are shifting the first load that we can from a timestamp perspective
            for i in range(len(idx_shiftable_loads)):
                
                if (difference > 0) and (load.newload[timestamp] <= peak_limit):
                    
                    # amount of load that will be shifted
                    load_shifted = load.newload[idx_shiftable_loads[i]] * flexibility_curve[idx_shiftable_loads[i]]
            
                    # A check to avoid the creation of new peaks in the consumption
                    if (load.newload[timestamp] + load_shifted) > peak_limit:
                        excess = load.newload[timestamp] + load_shifted - peak_limit
                        load_shifted = load_shifted - excess
            
                    # Then we shift the load
                    #print('Load that will be shifted:', load_shifted)
                    #print('Shifted from:', idx_shiftable_loads[i])
                    
                    # Adding the shifted load to the timestamp
                    load.newload[timestamp] = load.newload[timestamp] + load_shifted
                    
                    # And removing it from where it came from
                    load.newload[idx_shiftable_loads[i]] = load.newload[idx_shiftable_loads[i]] - load_shifted
                    
                    # And marking that the load was shifted
                    load.load_shift[timestamp].append(('to', idx_shiftable_loads[i])) # TO (1) timestamp FROM idx_shiftable_loads[i]
                    load.load_shift[idx_shiftable_loads[i]].append(('from', timestamp)) #FROM (-1) idx_shiftable_loads[i] TO timestamp
                    
                    # And the difference is now updated
                    difference = difference - load_shifted
                    #print('Updated difference:', difference)
         
                    
        # Seeing how the loads evolve in each timestep
        #plt.plot(load1.newload)

    # The load shifting is working as intended for improving self-consumption. Now we move to the case of the spot-price following
    
    else: 
        
        for load in load_list:
            #print('Load being processed!---')
            # How much load can we shift? We can't shift more than the flexibility limit of the load
            if (load.newload[timestamp] >= load.load[timestamp] - (flexibility_curve[timestamp] * load.load[timestamp])) and (spot_price_following == True):
                # The amount of load we can shift is then
                load_shifted = min(flexibility_curve[timestamp]*load.newload[timestamp], abs(difference))
                
                # We get which are the timestamps to where the load can be shifted
                times_to_shift = pairs[timestamp][1]
                
                # If we are doing it in real-time, we can only shift future loads. But that's not the case for now!
                # However, we can't change the past loads, so we can only shift the current load to a future timestamp
                #times_to_shift = [item for item in pairs[timestamp][1] if item > timestamp]
                
                
                
                # what is the current price
                current_price = price_data[timestamp]
                # We want to get the prices of all these timestamps to see when it is the lowest price in this interval
                prices_window = price_data[times_to_shift]
                
                # are there any times in the shifting window that would have smaller prices?
                smaller_prices = [price for price in prices_window if price < current_price]
                
                # If there are (i.e. smaller_prices is not empty)
                if len(smaller_prices) >= 1:
                    # if there are smaller prices in the window, where is the minimum?
                    index_of_min_price = np.where(prices_window == (min(smaller_prices)))[0][0]
                    
                    # we don't want to create any new peaks in the consumption
                    if (load.newload[times_to_shift[index_of_min_price]] + load_shifted) > peak_limit:
                        excess = load.newload[times_to_shift[index_of_min_price]] + load_shifted - peak_limit
                        load_shifted = load_shifted - excess
                    
                    #print('Load {:.2f} is being shifted from {} to {}'.format(load_shifted, timestamp, times_to_shift[index_of_min_price]))
                    
                    
                    # then we perform the load shift
                    load.newload[times_to_shift[index_of_min_price]] = load.newload[times_to_shift[index_of_min_price]] +\
                        load_shifted
                    
                    load.newload[timestamp] = load.newload[timestamp] - load_shifted
                    
                    # and let's mark that the load was shifted away 
                    # and the remaining load at timestamp is smaller (difference is negative)
                    
                    difference = difference + load_shifted
                    #print('Difference after spot-price shift: {:.2f}'.format(difference))
                    # we use (2) and (-2) to show the load shifts for spot-price following
                    load.load_shift[timestamp].append(('to', times_to_shift[index_of_min_price])) # shifted FROM (-2) timestamp TO times_to_shift[index_of_min_price]
                    load.load_shift[times_to_shift[index_of_min_price]].append(('from', timestamp)) #shifted TO (2) times_to_shift[index_of_min_price] FROM timestamp
            
            
        # Improve this plot, this is a good one!
        #plt.plot(load1.newload, color='k', linewidth=0.8, alpha=0.7)


    #print('Secondary check: Difference at this timestamp is {:.2f}'.format(difference))

    # Readjusting the total demand after the load shift
    total_demand_after_shift = np.sum([load.newload for load in load_list], axis=0)
            

#%%    
# And we now have the load shifts incorporated into the load objects and in the total demand
# We can now perform the microgrid operations


for timestamp in range(len(load_data)):
      
    ###### I: check excess generation after the load was shifted
    difference = gen_data[timestamp] - total_demand_after_shift[timestamp]
   
    ##### II: Load shifting already performed
    
    ##### III: Setting the BESS behaviour
    difference, BESS_parameters, BESS_SoC, BESS_io = mg.BESS_behaviour(difference, BESS_parameters, BESS_SoC, BESS_io, price_data, timestamp)
    
    
    ##### IV: Setting the EV behaviour   
    ### TO-DO: Create multiple EVs
    
    
    
    for EV in EV_list:
    
        if EV.plugged_array[timestamp] == 1:
            # the EV is available to be charged
            
            if (difference > 0):
                # we charge it with the excess generation
                EV_charge = min(difference, EV.cRate, EV.capacity - EV.SoC)
                EV.SoC = min(EV.SoC + EV_charge, EV.capacity)
                EV_discharge = 0
                
            else:
                # should we charge or discharge (v2g)?
                if (EV.V2G == True) and (EV.SoC > 
                    EV.discharge_threshold[0]*EV.capacity) and (price_data[timestamp] > price_threshold):
                    # It is worthwhile to discharge the EV to avoid paying a high price to grid imports
                    
                    EV_discharge = min(EV.cRate, abs(difference), (EV.SoC 
                                       - EV.discharge_threshold[1]*EV.capacity))
                    EV.SoC = max(EV.SoC - EV_discharge, 0)
                    EV_charge = 0
                    
                else:
                    # It's better to charge the EV or don't do anything with it
        
                    # we charge it by importing from the grid
                    EV_charge = min(EV.cRate, EV.capacity - EV.SoC)
                    EV.SoC = min(EV.SoC + EV_charge, EV.capacity)
                    EV_discharge = 0
            
            difference = difference - EV_charge + EV_discharge
            EV.day_disconnect = 0
            
        else:
            if EV.day_disconnect == 0:
                EV.SoC = EV.SoC - EV.battery_use[timestamp// int((24)*(60/minute_intervals))]*EV.SoC
                EV.day_disconnect = 1
            
            EV_charge = 0
            EV_discharge = 0
    
       
        EV.EV_io.append(EV_charge - EV_discharge) 
        EV.EV_SoC.append(EV.SoC)
    
    
    
    
    
      
    
    # difference, EV_parameters, EV_SoC, EV_io, EV_out_for_the_day = mg.EV_behaviour(timestamp, difference, EV_plugged, EV_parameters, 
    #                                                                                price_data, price_threshold, battery_usage_percentages, 
    #                                                                                minute_intervals, EV_out_for_the_day, EV_SoC, EV_io)


        
    
    # V: Setting the grid import/export behaviour
    grid_import, grid_export, grid_io = mg.grid_behaviour(difference, grid_io)
    
    # # V: Calculating the available flexibility in the microgrid at each timestamp
    
    # # BESS can up/down regulate by charging/discharging
    # BESS_up.append(min(BESS_parameters['SoC'], BESS_parameters['cRate']))
    # BESS_down.append(min(BESS_parameters['capacity'] - BESS_parameters['SoC'], BESS_parameters['cRate']))
    # # The BESS can also up/down regulate by just shifting the time which it charge/discharge
    # BESS_shift.append(BESS_io)
    
    
    # # Generation can only up-regulate by injecting power to the grid, and down-regulating by not injecting power to the grid (curtailing)
    # if difference > 0:
    #     gen_up.append(difference)
    #     gen_down.append(difference)
    # else:
    #     gen_up.append(0)
    #     gen_down.append(0)
    
    
    # # EV can up/down regulate by charging/discharging (if V2G is enabled) - and only if it is plugged!
    # if EV_plugged[timestamp] == 1:
    #     if (EV_parameters['V2G'] == True) and (EV_parameters['SoC'] > EV_parameters['discharge threshold'][0]*EV_parameters['capacity']):
    #         EV_up.append(min(EV_parameters['cRate'], (EV_parameters['SoC'] - EV_parameters['discharge threshold'][1]*EV_parameters['capacity'])))
    #     else:
    #         EV_up.append(0)
        
    #     EV_down.append(min(EV_parameters['capacity'] - EV_parameters['SoC'], EV_parameters['cRate']))
    #     EV_shift.append(EV_io)
    # else:
    #     EV_up.append(0); EV_down.append(0); EV_shift.append(0)
    
    
    # # Load flexibility
    
    # # up-regulation: reducing the load
    # # Was the load already shifted?
    # if load_shift[timestamp] < 0:
    #     load_up.append(0)
    # else:
    #     load_up.append(newload[timestamp] - flexibility_curve[timestamp]*newload[timestamp])
    
    
    # # down-regulation: increasing the load
    # # which timestamps the load can be moved FROM?
    # # Calculating the timestamp range (from flexibility curve) from which timestamps could be shifted
    # idx_shiftable_loads = [pair[0] for pair in pairs if timestamp in pair[1]]
    # # However, we can't alter the past, so we can only get indexes which are larget than timestamp
    # idx_shiftable_loads = [element for element in idx_shiftable_loads if element > timestamp]
    
    # maxload_timestamp = newload[timestamp]
    # for idx in idx_shiftable_loads:
    #     if load_shift[idx] >= 0:
    #         maxload_timestamp = maxload_timestamp + flexibility_curve[timestamp]*newload[idx]
    
    # load_down.append(maxload_timestamp)


    
    
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

# for col, data in microgrid_data.items():
#     microgrid_simulation[col] = data


# microgrid_simulation = pd.DataFrame({
#     #'Hour': range(1, len(load_data)+1),
#     'Original Load': load_data,
#     'Original Load2': load2.load,
#     'Total demand': total_demand,
#     'New Load1': load1.newload,
#     'New Load2': load2.newload,
#     'Total demand_shift': total_demand_after_shift,
#     'Load shift1': load1.load_shift,
#     'Load shift2': load2.load_shift,
#     'Generation': gen_data,
#     'BESS_SoC': BESS_SoC,
#     'BESS charge/discharge': BESS_io,
#     'EV_SoC': EV_SoC,
#     'EV charge/discharge': EV_io,
#     'Grid import/export': grid_io,
#     'Price data': price_data,
#     'EVPlug': EV_plugged,
#     #'Upreg': load_up,
#     #'Downreg': load_down
#     },
#     index=time_range)


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



