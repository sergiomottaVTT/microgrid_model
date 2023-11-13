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


import support_functions as fn
import mg_operations as mg
import mg_classes as cl

# %% DATA IMPORT
# Importing generation and load data, setting up the parameters of microgrid components
load_data = np.loadtxt(r'data/House1_LOADDATA_5000kWh-per-a.txt') #in kWh
load_data_2 = np.loadtxt(r'data/House1_LOADDATA_5500kWh-per-a.txt')
load_data_3 = np.loadtxt(r'data/House1_LOADDATA_6000kWh-per-a.txt')

gen_data = np.loadtxt(r'data/PV_pu_data.txt') #in pu
price_data = np.loadtxt(r'data/spotprice_2022.txt', usecols=2)

# How many days the simulation should run for
number_days = 365
# Setting the "sampling frequency" of the simulation: 15 minute intervals or 60 minute/hourly
minute_intervals = 15

# Load shifting is implemented
load_shifting = False
# Price to be considered as "expensive" and thus worth it to discharge BESS/EV/load shift
spot_price_following = False
price_threshold = 0.05

# whether we try to optimise for self-consumption and self-sufficiency
gen_shifting = False

# Setting up the parameters of the microgrid components
houses = 8
# If we want to have the same household loads repeated for N houses.

# PV system
PV_installed_capacity = 00.0 #kWp

gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_parameters = {'capacity': 0.0, #capacity in kWh
                   'cRate': 3.0, #charge/discharge rate in kW
                   'SoC': 00.0, # initial SoC
                   'Control': 'price_threshold', #the type of control for the battery, will discharge when threshold is above setpoint
                   'Control setpoint': price_threshold,
                   'Grid enabled': False,
                   'SoC threshold': 1
                   }

# EVs
EV_parameters = {'number': 4,   # How many EVs connected to the microgrid
                 'capacity': 40.0, #capacity
                 'SoC': 40.0,     # initial SoC
                 'cRate': 20.0 * (minute_intervals/60),   #charging rate of the charging station
                 'V2G': False,   #enabling V2G
                 'discharge threshold': (0.85, 0.6),    #can only be discharged if SoC > 85% capacity, down to 60% of capacity
                 }

# EV charging rate at 20kWh = 5kW per 15 minute interval
#scenarios = pd.read_excel('scenario_definition.xlsx', engine='openpyxl')


#%%
# Modifying the data to be at the right size and frequency
load_data, gen_data, price_data, final_time, time_range = fn.modify_data(load_data, gen_data, price_data, number_days, minute_intervals)
# And for the other load data:
load_data_2, _, _, _, _ = fn.modify_data(load_data_2, gen_data, price_data, number_days, minute_intervals)
load_data_3, _, _, _, _ = fn.modify_data(load_data_3, gen_data, price_data, number_days, minute_intervals)




# %% Fixed-price electricity instead of spot-price

# Using HELEN's latest prices
#https://www.helen.fi/en/electricity/electricity-products-and-prices

# helen_price_data = np.ones_like(price_data)
# price_data = 0.0899 * helen_price_data


# %% Implementing some object-oriented programming for the first time in Python!

# If we create equal loads:
# load_list = []

# for _ in range(houses):
#     load_gen = cl.Load(load_data, load_data, load_shifting)
#     load_gen.define_flexibility(0.15, 2, number_days, minute_intervals, plot=False)
#     load_list.append(load_gen)


# If we want to create individual loads:

# 2 houses consuming 5000kWh per year
load1 = cl.Load(load_data, load_data, load_shifting)
load2 = cl.Load(load_data, load_data, load_shifting)
# 4 houses consuming 5500kWh per year
load3 = cl.Load(load_data_2, load_data_2, load_shifting)
load4 = cl.Load(load_data_2, load_data_2, load_shifting)
load5 = cl.Load(load_data_2, load_data_2, load_shifting)
load6 = cl.Load(load_data_2, load_data_2, load_shifting)
# 2 houses consuming 6000kWh per year
load7 = cl.Load(load_data_3, load_data_3, load_shifting)
load8 = cl.Load(load_data_3, load_data_3, load_shifting)

# Creating a list of all our loads
load_list = [load1, load2, load3, load4, load5, load6, load7, load8]

# Setting the load flexibility behaviour
for load in load_list:
    load.define_flexibility(0.15, 4, number_days, minute_intervals, plot=False)



# The total load of the microgrid and the total load considering also BESS and EVs as negative loads
total_demand = np.sum([load.load for load in load_list], axis=0)

total_demand_after_shift = np.sum([load.newload for load in load_list], axis=0)


# %% Transforming the EVs into objects

EV_list = []


### TO-DO: EV behaviour can be changed so they remain unplugged for longer in some days!
# The consumption from the EV battery should be proportional to how long the EV was disconnected!


# for _ in range(EV_parameters['number']):
#     gen_ev = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], True)
#     gen_ev.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False)
#     EV_list.append(gen_ev)
    
# Creating EVs and Setting EV behaviour
EV1 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV1.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False)

EV2 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV2.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=True)

EV3 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV3.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=True)

EV4 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV4.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=True)

EV_list = [EV1, EV2, EV3, EV4]


#%% MICROGRID OPERATION

### TO-DO: Power-based tariffs!
### TO-DO: Calculate available flexibility at each timestamp
### TO-DO: Monte Carlo simulation to calculate the flexibility activation in frequency markets
### TO-DO: Preliminary load shifts; then real-time load shifts following market bidding

peak_limit = np.max(load_list[0].newload)

# Performing the day-ahead load shifts
# "day-ahead" meaning that we perform load shifts from past and future timestamps, with a full-picture, as 
# opposed to being able to shift only future loads (and not past loads)

load_list, total_demand_after_shift, BESS_SoC, BESS_io, EV_list, grid_io = mg.mg_day_ahead_operation(load_list, BESS_parameters, 
                                                                                                     EV_list, gen_data, total_demand_after_shift, 
                                                                                                     price_data, peak_limit, price_threshold, 
                                                                                                     minute_intervals, gen_shifting, spot_price_following)



# %%  Assigning values to a dataframe for easier inspection
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


microgrid_simulation = pd.DataFrame(microgrid_data, index=pd.to_datetime(time_range, format='%d-%m-%Y %H:%M'))


fn.check_mg(microgrid_simulation)

# %% Evaluating results

# import matplotlib.dates as md

# fig, ax = plt.subplots()
# ax.plot(microgrid_simulation.index, microgrid_simulation['Total demand'], label='Original demand')
# ax.plot(microgrid_simulation.index, microgrid_simulation['Total demand_shift'], label='Demand after shift')
# ax.plot(microgrid_simulation.index, microgrid_simulation['Grid import/export'], label='Grid import/export')
# ax.set_title('Original demand x Shifted demand')
# ax.set_xlabel('Time')
# ax.set_ylabel('Energy (kWh)')
# ax.xaxis.set_major_locator(md.HourLocator(interval=5))
# ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
# ax.legend()


# %% Evaluating the KPIs and economic benefit

KPI_scss, KPI_econ = fn.mg_eval(microgrid_simulation, minute_intervals)


# %% Calculating the average loads

time_intervals = pd.date_range(start='00:00', end='23:45', freq='15T').strftime('%H:%M')
daily_values = pd.DataFrame(index=time_intervals)
hourly_values = []

# for idx, row in daily_values.iterrows():
#     mask = (microgrid_simulation.index.hour == idx.hour) & (microgrid_simulation.index.minute == idx.minute)
#     hourly_values.append(microgrid_simulation.loc[mask, 'Total demand_shift'].values)



for hour in range(24):
    for minute in [00, 15, 30, 45]:
        mask = (microgrid_simulation.index.hour == hour) & (microgrid_simulation.index.minute == minute)
        #daily_values.loc[hour:minute, 'Values'] = microgrid_simulation.loc[mask, 'Total demand_shift'].values
        #daily_values['Values'] = microgrid_simulation.loc[mask, 'Total demand_shift'].values
        hourly_values.append(microgrid_simulation.loc[mask, 'Total demand_shift'].values)

daily_values['Values'] = hourly_values

# %% 
# fig, ax = plt.subplots()
# microgrid_simulation.boxplot(column='Total demand_shift', by=microgrid_simulation.index.strftime('%H:%M').rename('Hour:Minute'), ax=ax)
# ax.set_title("Distribution of Values per 15-Minute Interval")
# ax.set_ylabel("Demand")
# plt.suptitle('')  # Remove the default title
# plt.show()



# %% Saving results
# import pickle

# with open(column+'.pkl', 'wb') as file:
#     pickle.dump([microgrid_simulation, KPI_scss, KPI_econ], file)









