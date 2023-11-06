# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:26:10 2023

@author: sdsergio

Defining custom functions for Microgrid simulations
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta


# %% Array interpolation for transforming arrays from hourly values to X minute intervals

# Performing the interpolation for 15 minute intervals:
def min_intervals(array, interval_mins):
    """
    Performing a linear interpolation between hourly values to generate arrays with minute intervals

    Parameters
    ----------
    array :np.array
        The original array to be transformed.
    interval_mins : int
        Minute intervals, e.g. 15 = 15 minute intervals.

    Returns
    -------
    interpolated_array : np.array
        The interpolated array with X minute intervals instead of hourly values.

    """
    # Defining how many indices will there be for this new array with 'interval_mins' minute intervals
    indices = np.arange((60/interval_mins)*len(array))
    # Performing the interpolation
    interpolated_array = np.interp(indices, np.arange(len(array))*4, array)
    
    return interpolated_array


# %% Data modification


def modify_data(load_data, gen_data, price_data, number_days, minute_intervals, plotting=False):
    """
    Function to make the data modifications: array length, interpolate for different time intervals.

    Parameters
    ----------
    load_data : array
        Array with the original load consumption.
    gen_data : array
        Array with the original generation data.
    price_data : array
        Array with the original spot price data.
    number_days : int
        How many days are considered in the simulation.
    minute_intervals : int
        Sampling frequency of the simulation.
    plotting : Bool, optional
        Switch to output plots of the arrays after their modification. The default is True.

    Returns
    -------
    load_data : array
        modified load data array.
    gen_data : array
        modified generation data array.
    price_data : array
        modified price data array.
    final_time : float
        final timestamp (23, 23.75) to be used for the EV behaviour.
    time_range : datetime index
        index range of datetime values to be used as the microgrid dataframe index.

    """
    # Cutting the raw data to be on the right size (amount of days)
    # We start from the second day just because the load behaves a bit strangely in the first hours
    load_data = load_data[24:24*(number_days+1)]
    gen_data = gen_data[24:24*(number_days+1)]
    
    # Spot price data we have starts from 7 in the evening in the last day
    price_data = price_data[7:(24*number_days)+7]/1000 #to have in EUR/kWh
    
    # Making the interpolation for 15-minute intervals
    if minute_intervals != 60:
        final_time = 23.75
        
        load_data = min_intervals(load_data, minute_intervals)
        gen_data = min_intervals(gen_data, minute_intervals)
        price_data = min_intervals(price_data, minute_intervals)
        
    else:
        final_time = 23
    
    # Setting the dataframe time intervals
    start_time = "2022-01-01"
    
    if minute_intervals == 60:
        time_range = pd.date_range(start=start_time, periods=24*number_days, freq='H')
    elif minute_intervals == 15:
        time_range = pd.date_range(start=start_time, periods=24*4*number_days, freq='15T')
    
    time_range = time_range.strftime("%d-%m-%Y %H:%M")
    
    if plotting == True:
    #### If we want to see how the load, generation, and prices look like
        plt.figure()
        plt.plot(load_data, label='load')
        plt.plot(gen_data, label='generation')
        plt.title('Load and generation data')
        plt.xlabel('time (h)')
        plt.ylabel('kWh')
        plt.legend()
        
        plt.figure()
        plt.plot(price_data, label='spot price data in EUR/kWh')
        plt.title('Spot-price data')
        plt.xlabel('time (h)')
        plt.ylabel('EUR/kWh')
        plt.legend()
        
        
    return load_data, gen_data, price_data, final_time, time_range



# %% Result evaluation


def mg_eval(microgrid_simulation, minute_intervals, sc=True, ss=True, econ=True):
    """
    Function to evaluate the KPIs and economic behaviour of the microgrid (cost savings due to load shifting and self-consumption)

    Parameters
    ----------
    microgrid_simulation : Dataframe
        Dataframe with the microgrid operation per timestamp.
    minute_intervals : Int
        Frequency of the timestamps.
    sc : Bool, optional
        Bool to define whether we want to see the output of self-consumption calculations. The default is True.
    ss : Bool, optional
        Bool to define whether we want to see the output of self-sufficiency calculations. The default is True.
    econ : Bool, optional
        Bool to define whether we want to see the output of economic calculations. The default is True.

    Returns
    -------
    KPI_scss : Dict
        Dictionary with the calculated values for the KPIs.
    KPI_econ : Dict
        Dictionary with the calculated values for economic costs and income.
    """
    # self-consumption
    
    # Since we have many energy storage assets that are charged/discharged with locally generated energy, the caluclation of self-consumption and self-sufficiency
    # is slighly more challenging. We must then consider the grid imports/exports to ease this calculation.
    
    #self_consumption = (total generated - total exported)/total generated
    
    total_gen = microgrid_simulation['Generation'].sum() * minute_intervals/60
    total_export = microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] > 0].sum() * minute_intervals/60
    
    self_consumption = (total_gen - total_export)/total_gen

    if sc == True:
        print('The total self-consumption of the microgrid was {:.2f}%'.format(self_consumption * 100))
        print('The total exports to the grid were {:.2f} kWh'.format(total_export))
        print('And the total generation was {:.2f} kWh'.format(microgrid_simulation['Generation'].sum()))
    
    total_consumption_loads = microgrid_simulation['Total demand_shift'].sum() * minute_intervals/60
    total_consumption_BESS = microgrid_simulation['BESS charge/discharge'].sum() * minute_intervals/60
    total_consumption_EVs = np.sum(microgrid_simulation[[col for col in microgrid_simulation.columns if 'I/O' in col]].sum()) * minute_intervals/60
    
    total_consumption = total_consumption_loads + total_consumption_BESS + total_consumption_EVs
    
    total_import = abs(microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] < 0].sum()) * minute_intervals/60
    
    
    self_sufficiency = (total_consumption - total_import)/total_consumption
    
    if ss == True:
        print('The total self-sufficiency of the microgrid was {:.2f}%'.format(self_sufficiency * 100))
        print('The total load consumed was {:.2f} kWh'.format(total_consumption))
        print('The total imports from the grid were {:.2f} kWh'.format(total_import))
    
    KPI_scss = {'total_gen': total_gen,
          'total_export': total_export,
          'consumption_loads': total_consumption_loads,
          'consumption_BESS': total_consumption_BESS,
          'consumption_EV': total_consumption_EVs,
          'consumption': total_consumption,
          'import': total_import,
          'export': total_export,
          'self_consumption': self_consumption,
          'self-sufficiency': self_sufficiency}
    
    # Calculating the costs
    # We should calculate costs in a generic way
    
    # Costs with the loads (regardless of local generation, storage, etc.)
    
    load_costs = np.sum(microgrid_simulation['Total demand_shift'] * microgrid_simulation['Price data']) * minute_intervals/60
    
    # BESS charging costs
    BESS_costs = np.sum(microgrid_simulation['BESS charge/discharge'][microgrid_simulation['BESS charge/discharge'] > 0] 
                        * microgrid_simulation['Price data']) * minute_intervals/60
    
    # EV charging costs
    EV_costs = np.sum(microgrid_simulation[[col for col in microgrid_simulation.columns if 'I/O' in col]]
                                                     [microgrid_simulation[[col for col in microgrid_simulation.columns if 'I/O' in col]] > 0].sum(axis=1) \
                                                         * microgrid_simulation['Price data']) * minute_intervals/60
    
    # Grid import costs (should be the sum of all these)
    grid_import_costs = abs(np.sum(microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] < 0 ] 
                               * microgrid_simulation['Price data'])) * minute_intervals/60
    
    # Grid export income
    grid_export_income = abs(np.sum(microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] > 0 ] 
                               * microgrid_simulation['Price data'])) * minute_intervals/60
    
    if econ == True:
        print('\n## Economic evaluation ##\n')
        print('Load costs: {:.2f} EUR'.format(load_costs))
        print('BESS charging costs: {:.2f} EUR'.format(BESS_costs))
        print('EV charging costs: {:.2f} EUR'.format(EV_costs))
        print('Grid import costs: {:.2f} EUR'.format(grid_import_costs))
        print('Grid export income: {:.2f} EUR'.format(grid_export_income))

    KPI_econ = {'load costs': load_costs,
                'BESS costs': BESS_costs,
                'EV costs': EV_costs,
                'grid import': grid_import_costs,
                'grid export': grid_export_income}

    return KPI_scss, KPI_econ


