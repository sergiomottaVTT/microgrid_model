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
    
    time_range = time_range.strftime("%d-%m %H:%M")
    
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

def result_eval(microgrid_simulation, minute_intervals, show_costs=True, show_energy=True):
    """
    Function to calculate and show summary results on economic analysis and energy consumption.

    Parameters
    ----------
    microgrid_simulation : dataframe
        Dataframe with the microgrid simulation results per timestamp.
    minute_intervals : int
        The time frequency of the array, 1-hour time intervals or 15-minute time intervals.
    show_costs : Bool, optional
        Whether the function should output the printed costs. The default is True.
    show_energy : Bool, optional
        Whether the function should output the energy calculations. The default is True.

    Returns
    -------
    savings : float
        How much money was saved in energy imports.
    benefit : float
        Total economic benefit: savings + energy exports.
    self_sufficiency: float
        How much of the total energy needs are supplied by the microgrid assets versus the imported from the grid.
    self_consumption: float
        How much of the energy produced locally is used against how much is exported.

    """ 
    # Costs
    original_cost = np.sum(microgrid_simulation['Original Load'] * microgrid_simulation['Price data']) / (60 / minute_intervals)
    
    grid_io_prices = microgrid_simulation['Grid import/export'] * microgrid_simulation['Price data']
    import_costs = abs(np.sum(grid_io_prices[grid_io_prices < 0]) / (60 / minute_intervals))
    export_income = np.sum(grid_io_prices[grid_io_prices >= 0]) / (60 / minute_intervals)
    
    savings = original_cost - import_costs
    benefit = savings + export_income
    
    # Energy use
    original_energy_grid = np.sum(microgrid_simulation['Original Load']) / (60 / minute_intervals)
    new_energy_grid = abs(np.sum(microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] < 0]) / ( 60 / minute_intervals))
    exported_energy = np.sum(microgrid_simulation['Grid import/export'][microgrid_simulation['Grid import/export'] >= 0]) / ( 60 / minute_intervals)
    
    original_consumed_energy_excess_gen = np.sum(microgrid_simulation['Original Load'].loc[microgrid_simulation['Generation'] 
                                                                                           > microgrid_simulation['Original Load']]) / (60 / minute_intervals)
    energy_self_consumed = np.sum(microgrid_simulation['New Load'].loc[microgrid_simulation['Generation'] 
                                                                                           > microgrid_simulation['Original Load']]) / (60 / minute_intervals)
    
    # Other KPIs
    self_sufficiency = 1 - abs(np.sum(microgrid_simulation['Grid import/export']
                                      [microgrid_simulation['Grid import/export'] < 0])) / np.sum(microgrid_simulation['New Load'])
    self_consumption = 1 - np.sum(microgrid_simulation['Grid import/export']
                                  [microgrid_simulation['Grid import/export'] >= 0])/np.sum(microgrid_simulation['Generation'])
    
    ####### Printing
    if show_costs == True:
        print('Original costs: {:.2f} EUR'.format(original_cost))
        print('New import costs: {:.2f} EUR'.format(import_costs))
        print('Savings due to microgrid: {:.2f} EUR'.format(savings))
        
        print('Export income: {:.2f} EUR'.format(export_income))
        print('Total benefit: {:.2f} EUR'.format(benefit))

    if show_energy == True:
        print('Original energy use from grid: {:.2f} kWh'.format(original_energy_grid))
        print('New energy use from grid: {:.2f} kWh'.format(new_energy_grid))
        print('Exported energy to grid: {:.2f} kWh'.format(exported_energy))
        
        print('Original energy consumed during the time that we have excess gen: {:.2f} kWh'.format(original_consumed_energy_excess_gen))
        print('Energy self-consumed: {:.2f} kWh'.format(energy_self_consumed))

        print('Self-sufficiency: {:.2f}%'.format(self_sufficiency * 100))
        print('Self-consumption: {:.2f}%'.format(self_consumption * 100))

    return savings, benefit, self_sufficiency, self_consumption
