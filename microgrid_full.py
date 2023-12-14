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
price_data = np.loadtxt(r'data/price_average.txt')

# How many days the simulation should run for
number_days = 365
# Setting the "sampling frequency" of the simulation: 15 minute intervals or 60 minute/hourly
minute_intervals = 15

# Load shifting is implemented
load_shifting = False
# Price to be considered as "expensive" and thus worth it to discharge BESS/EV/load shift
spot_price_following = False
# Below this value, the BESS and EV also consider it cheap enough to charge.
price_threshold = 1.08

# whether we try to optimise for self-consumption and self-sufficiency
gen_shifting = False

# whether we want fixed-price or spot-price (false)
fixed = False

# Setting up the parameters of the microgrid components
houses = 20
# If we want to have the same household loads repeated for N houses.

# Setting up the parameters for the house flexibility values
flex_value = 0.1
flex_time = 4 #in timestamps, 4 = 1 hour


# PV system
PV_installed_capacity = 50.0 #kWp

gen_data = gen_data * PV_installed_capacity #gen_data in kW

# BESS
BESS_parameters = {'capacity': 30.0, #capacity in kWh
                   'cRate': 3.0, #charge/discharge rate in kW
                   'Initial SoC': 30.0, # initial SoC
                   'SoC': 30.0, # variable SoC
                   'Control': 'load_threshold',#'price_threshold', #the type of control for the battery, will discharge when threshold is above setpoint
                   'Control setpoint': 8.00,#price_threshold,
                   'Price threshold': price_threshold,
                   'Grid enabled': False,
                   'SoC threshold': 1
                   }

# EVs
EV_parameters = {'number': 4,   # How many EVs connected to the microgrid
                 'capacity': 40.0, #capacity
                 'SoC': 40.0,     # initial SoC
                 'cRate': 20.0 * (minute_intervals/60),   #charging rate of the charging station
                 'V2G': True,   #enabling V2G
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

if fixed == True:
    helen_price_data = np.ones_like(price_data)
    price_data = 0.0899 * helen_price_data
    
    
# Distribution prices
# Fixed price of distribution is not relevant, as it will be the same anyway. But we have the distribution fee in cents/kWh

distribution_prices = 0.05 * np.ones_like(price_data) #EUR/kWh

# %% Creating loads

#If we create equal loads:
load_list = []

for _ in range(houses):
    load_gen = cl.Load(load_data, load_data, load_shifting)
    load_gen.define_flexibility(flex_value, flex_time, number_days, minute_intervals, plot=False)
    load_list.append(load_gen)


# # If we want to create individual loads:

# # 2 houses consuming 5000kWh per year
# load1 = cl.Load(load_data, load_data, load_shifting)
# load2 = cl.Load(load_data, load_data, load_shifting)
# # 4 houses consuming 5500kWh per year
# load3 = cl.Load(load_data_2, load_data_2, load_shifting)
# load4 = cl.Load(load_data_2, load_data_2, load_shifting)
# load5 = cl.Load(load_data_2, load_data_2, load_shifting)
# load6 = cl.Load(load_data_2, load_data_2, load_shifting)
# # 2 houses consuming 6000kWh per year
# load7 = cl.Load(load_data_3, load_data_3, load_shifting)
# load8 = cl.Load(load_data_3, load_data_3, load_shifting)

# # Creating a list of all our loads
# load_list = [load1, load2, load3, load4, load5, load6, load7, load8]

# # Setting the load flexibility behaviour
# for load in load_list:
#     load.define_flexibility(0.10, 4, number_days, minute_intervals, plot=False)



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
EV2.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=False)

EV3 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV3.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=False)

EV4 = cl.EV(EV_parameters['capacity'], EV_parameters['cRate'], EV_parameters['SoC'], EV_parameters['discharge threshold'], EV_parameters['V2G'])
EV4.set_EV_behaviour(number_days, final_time, minute_intervals, load_data, plot=False, random=False)

EV_list = [EV1, EV2, EV3, EV4]


#%% MICROGRID OPERATION

### TO-DO: Power-based tariffs!
### TO-DO: Calculate available flexibility at each timestamp
### TO-DO: Monte Carlo simulation to calculate the flexibility activation in frequency markets
### TO-DO: Preliminary load shifts; then real-time load shifts following market bidding
### TO-DO: CHECK IF THE LOAD SHIFTING IS BEING LIMITED! IT SHOULDNT BE POSSIBLE TO GO BELOW MINLOAD!!!


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


microgrid_simulation['Distribution prices'] = distribution_prices

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


fn.printing_scenario(number_days, minute_intervals, load_shifting, spot_price_following, gen_shifting, fixed, houses, BESS_parameters, EV_parameters)

KPI_scss, KPI_econ = fn.mg_eval(microgrid_simulation, minute_intervals)


# %% Checking the average load shift performed

mg_loadshift = microgrid_simulation[['Total demand', 'Total demand_shift']].copy()

mg_loadshift['Total diff'] = mg_loadshift['Total demand_shift'] - mg_loadshift['Total demand']

# Getting only the load shifted away (it should be shifted to somewhere, as the sum in diff = 0)
mg_shiftedaway = pd.DataFrame(mg_loadshift['Total diff'][mg_loadshift['Total diff'] > 0])
mg_shiftedaway.index = pd.to_datetime(mg_shiftedaway.index, format='%Y%m%d %H:%M')

print('Total amount of load shifted in a year: {:.2f} kWh'.format(mg_shiftedaway['Total diff'].sum()))

mg_shiftedaway['Day'] = mg_shiftedaway.index.day
mg_shiftedaway['Month'] = mg_shiftedaway.index.month

print('Average load shift per timestamp: {:.2f} kWh'.format(mg_shiftedaway['Total diff'].mean()))

# grouping by month and day to get how much load was shifted ON AVERAGE each on each timestamp of every day
mg_avgdailyshift = mg_shiftedaway.groupby(['Month', 'Day'])['Total diff'].mean().reset_index()
print('Average daily load shifted: {:.2f} kWh'.format(mg_avgdailyshift['Total diff'].mean()))

mg_totaldailyshift = mg_shiftedaway.groupby(['Month', 'Day'])['Total diff'].sum().reset_index()
print('Total daily load shifted: {:.2f} kWh'.format(mg_totaldailyshift['Total diff'].mean()))

#Calculating the % of daily load

mg_dailyload = microgrid_simulation[['Total demand', 'Total demand_shift']].copy()
mg_dailyload.index = pd.to_datetime(mg_dailyload.index, format='%Y%m%d %H:%M')
mg_dailyload['Day'] = mg_dailyload.index.day
mg_dailyload['Month'] = mg_dailyload.index.month

mg_dailyload = mg_dailyload.groupby(['Month', 'Day'])['Total demand_shift'].sum().reset_index()
mg_dailyload['Total demand_shift'] = mg_dailyload['Total demand_shift'] * minute_intervals/60


percentage = mg_totaldailyshift['Total diff']/mg_dailyload['Total demand_shift']
print('Average percentage of load shifted each day: {:.2f}%'.format(percentage.mean()*100))





# %% Saving results 

#microgrid_simulation.to_pickle(r'data/results/01flex_1h_20houses_base.pkl')

# %% Calculating the flexibility availability

# # we want to calculate how much up- and down-regulation is available from the Energy Community at each time-step.
# # i.e., how much flexibility the SDN can aggregate and offer to frequency markets.

# # We want to provide up-regulation, which is adding energy or reducing the load to increase the frequency; and down-regulation, which is 
# # reducing the energy injected or increasing the load to reduce the frequency. Each asset in the microgrid will then behave differently in this context.

# #### LOADS

# # the maximum load possible at each time-step is calculated by the method Load.define_flexibility, and is represented in the parameter Load.maxload
# # The maximum load is calculated as the sum of all possible load shifts from adjacent time-steps.
# # Similarly, the minimum load is also calculated by the same method and expressed in Load.minload. It is the reduction of (%) of flexibility in each time-step.

# # For the first two days and load 1, we have
# plt.figure()
# plt.plot(load1.maxload[0:192], linestyle='--', linewidth=0.7, label='maxload')
# plt.plot(load1.load[0:192], label='Original load values')
# plt.plot(load1.newload[0:192], label='Shifted load')
# plt.plot(load1.minload[0:192], linestyle='--', linewidth=0.7, label='minload')
# plt.legend()

# # In this case, the maximum load possible is much higher than the load peak, so we'll certainly have extra peaks in the distribution grid if we shift 
# # loads up until maxload of a time-step.

# # The up-regulation potential from the load is calculated by the difference between the load at time-step (t) and the minimum load at time-step (t), i.e.,
# # how much load is there still to be shifted away from this time-step.

# up_reg_load = load1.newload - load1.minload
# # And the down-regulation potential is the difference between the maximum possible load and the load at time (t), i.e., how much load can still be shifted to
# # this time-step.
# down_reg_load = load1.maxload - load1.newload

# #%%
# #### BESS

# # For the BESS, up-regulation means delaying the charge (when charging from the grid), i.e. reducing its load, or discharging (when idle)

# # When we have grid imports (<0), the upregulation from the BESS charging would be just its capacity to reduce load (its I/O). When we export to the grid, 
# # there's no upregulation possible from the charging of the battery, as it's already charging at its limit because we prioritise BESS charging over grid export.
 
# up_reg_bess_ch = np.where(microgrid_simulation['Grid import/export'].values < 0, microgrid_simulation['BESS charge/discharge'].values, 0)

# # However, for up-regulation, we only want to consider the times when the BESS is charging, so we calculate how much charging can be delayed
# up_reg_bess_ch = np.where(up_reg_bess_ch > 0, up_reg_bess_ch, 0) 

# # We can also provide up-regulation when the BESS is discharging or idle.
# # The discharge up-regulation is the difference between what CAN be discharged (min(SoC, cRate)) and what IS being discharged (negative BESS charge/discharge)

# up_reg_bess_disch_capacity = np.min([microgrid_simulation['BESS_SoC'].values, BESS_parameters['cRate']*np.ones_like(microgrid_simulation['BESS_SoC'].values)], axis=0)

# # the BESS discharge that is happening is
# up_reg_bess_disch_happening = np.where(microgrid_simulation['BESS charge/discharge'] < 0, microgrid_simulation['BESS charge/discharge'], 0)

# # However we must keep in mind that the BESS SoC is AT THE END of the timestamp, whereas the BESS I/O is DURING the timestamp.
# # This means that for instance, the BESS was charged from 0kWh to 1kWh at between 11:00 - 11:15, the I/O column will have 
# # 1kWh at 11:00 timestmap, with a 0kWh SoC; and at the timestamp 11:15 its SoC is now 1kWh. 
# # Thus, we must "jump" the SoC values one index later to reflect the flexibility capacity that is available at that timestamp.
# up_reg_bess_disch_capacity = np.roll(up_reg_bess_disch_capacity, 1)
# up_reg_bess_disch_capacity[0] = min(BESS_parameters['Initial SoC'], BESS_parameters['cRate'])

# # Now we can calculate how much available flexibility is there at each timestamp

# up_reg_bess_disch = up_reg_bess_disch_capacity - abs(up_reg_bess_disch_happening)


# # To check it

# up_reg_bess = up_reg_bess_ch + up_reg_bess_disch

# # Now we perform the dowm-regulation for the BESS. This means calculating how much extra load (charging or stopped discharging) can be added in each timestamp.

# # How much can we increase charging
# down_reg_bess_ch_capacity = np.min([BESS_parameters['capacity'] - microgrid_simulation['BESS_SoC'].values, 
#                                     BESS_parameters['cRate']*np.ones_like(microgrid_simulation['BESS_SoC'].values)], axis=0)

# down_reg_bess_ch_happening = np.where(microgrid_simulation['BESS charge/discharge'] > 0, microgrid_simulation['BESS charge/discharge'], 0)

# down_reg_bess_ch_capacity = np.roll(down_reg_bess_ch_capacity, 1)
# down_reg_bess_ch_capacity[0] = min(BESS_parameters['capacity'] - BESS_parameters['Initial SoC'], BESS_parameters['cRate'])

# down_reg_bess_ch = down_reg_bess_ch_capacity + abs(down_reg_bess_ch_happening)



# # How much can we reduce discharging (thus increasing the overall load) - it can be the WHOLE discharging when we have grid imports.
# down_reg_bess_disch = np.where(microgrid_simulation['Grid import/export'].values < 0, microgrid_simulation['BESS charge/discharge'].values, 0)
# down_reg_bess_disch = np.where(down_reg_bess_disch < 0, down_reg_bess_disch, 0)


# down_reg_bess = down_reg_bess_ch + abs(down_reg_bess_disch)


# BESS_regulation = pd.DataFrame()

# BESS_regulation['BESS_SoC'] = microgrid_simulation['BESS_SoC']
# BESS_regulation['BESS_io'] = microgrid_simulation['BESS charge/discharge']

# BESS_regulation['up_charging avoidable']
# BESS_regulation['up_discharge cap']
# BESS_regulation['up_reg']
# BESS_regulation['down_discharging avoidable']
# BESS_regulation['down_charging cap']
# BESS_regulation['down_reg']



# BESS_regulation = pd.DataFrame({'UP - Discharge Capacity': up_reg_bess_disch_capacity, 
#                                 'UP - Discharging happening': up_reg_bess_disch_happening,
#                                 'UP - Discharge Potential': up_reg_bess_disch})

# BESS_regulation['DOWN - regulation potential'] = down_reg_bess
# BESS_regulation['UP - Charging happening avoidable'] = up_reg_bess_ch
# BESS_regulation['UP - regulation potential'] = up_reg_bess
# BESS_regulation['DOWN - Discharging happening avoidable'] = down_reg_bess_disch
# BESS_regulation['DOWN - Charge Capacity'] = down_reg_bess_ch_capacity
# BESS_regulation['DOWN - Charging Occurring'] = down_reg_bess_ch_happening
# BESS_regulation['DOWN - Charging Potential'] = down_reg_bess_ch




# up_reg_bess_disch = np.min([microgrid_simulation['BESS_SoC'].values, BESS_parameters['cRate']*np.ones_like(microgrid_simulation['BESS_SoC'].values)], axis=0)
# up_reg_bess_disch = up_reg_bess_disch - abs(np.where(microgrid_simulation['BESS charge/discharge'] < 0, microgrid_simulation['BESS charge/discharge'], 0))

# if BESS_parameters['Grid Enabled'] == True:
#     # the BESS_io will be positive (charge) also in times when there's no excess generation 
#     if BESS_io > 0 and : # it's charging
#         BESS_ch_upreg = BESS_io
# in each time-step
# check if it is charging, discharging, or idle
# if charging, check if it is from the grid, and if it is, it can be delayed
#up_reg_BESS = BESS_charging
# if discharging or idle, up-regulation capacity is limited only by c-Rate
#up_reg_BESS = min(cRate, capacity)


# %% Calculating the average loads

# time_intervals = pd.date_range(start='00:00', end='23:45', freq='15T').strftime('%H:%M')
# daily_values = pd.DataFrame(index=time_intervals)
# hourly_values = []

# # for idx, row in daily_values.iterrows():
# #     mask = (microgrid_simulation.index.hour == idx.hour) & (microgrid_simulation.index.minute == idx.minute)
# #     hourly_values.append(microgrid_simulation.loc[mask, 'Total demand_shift'].values)



# for hour in range(24):
#     for minute in [00, 15, 30, 45]:
#         mask = (microgrid_simulation.index.hour == hour) & (microgrid_simulation.index.minute == minute)
#         #daily_values.loc[hour:minute, 'Values'] = microgrid_simulation.loc[mask, 'Total demand_shift'].values
#         #daily_values['Values'] = microgrid_simulation.loc[mask, 'Total demand_shift'].values
#         hourly_values.append(microgrid_simulation.loc[mask, 'Total demand_shift'].values)

# daily_values['Values'] = hourly_values

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






