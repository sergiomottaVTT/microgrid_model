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



# %% Minimum and Maximum loads

def min_max_loads(load_data, flexibility_curve, flexibility_window, **kwargs):
    minload, maxload, addload, addload_future = np.zeros_like(load_data), np.zeros_like(load_data), np.zeros_like(load_data), np.zeros_like(load_data)
    plot = kwargs.get('plot')
    
    # Calculating the minimum possible load
    for timestamp in range(len(load_data)):
        # This is the lowest possible load that can occur at this time
        minload[timestamp] = load_data[timestamp] - flexibility_curve[timestamp]*load_data[timestamp]
    
    # Calculating the maximum possible load
    pairs = []
    for timestamp in range(len(load_data)):
        # we iterate over each hourly value of the load
        # how much load will be added to previous and subsequent hours
        load_shift = flexibility_curve[timestamp] * load_data[timestamp]
        
        # Getting the indices in the load vector to be changed:
        idx_neg = timestamp - flexibility_window[timestamp]     
        idx_pos = timestamp + flexibility_window[timestamp]
        
        # There is no rollout as we have continuous data
        if idx_neg <0: idx_neg = 0
        if idx_pos+1 > len(load_data): idx_pos = len(load_data)-1
        
        # To create the list with symmetrical indices, excluding the central timestamp (where the load will be shifted)
        previous = list(range(idx_neg, timestamp))
        posterior = list(range(timestamp+1, idx_pos+1))
    
        symmetrical_indexes = previous + posterior
        
        # if we consider only the future indices, because we can't change the past timestamps
        future_indexes_only = posterior
        
        # just to save the hours and the indexes
        pairs.append((timestamp, symmetrical_indexes))
        
        for indices in symmetrical_indexes:
            addload[indices] = addload[indices] + load_shift
    
    
        for indices in future_indexes_only:
            addload_future[indices] = addload_future[indices] + load_shift
    
    # the maximum load possible is the combination of the "normal" load and the potential for all load shifting from nearby hours
    maxload = load_data + addload
    
    # the maximum load possible when considering only future load shifts is
    maxload_future = load_data + addload_future
    
    
    if plot == True:
        plt.figure()
        plt.plot(load_data, label='Average load')
        plt.plot(minload, linestyle='--', label='Minimum load')
        plt.plot(maxload, linestyle='--', label='Maximum load')
        plt.title('Min, Max and Avg load curves')
        plt.legend()
        
    return pairs, minload, maxload, maxload_future

# %% Defining flexibility interval and window

def define_flexibility(number_days, minute_intervals, load_data, plot=False):
    
    # Each timestamp has a different amount of flexibility, based on people's willingness to modify their consumption behaviour (% of the load)
    # flexibility_curve = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, # early morning, 00:00 to 05:00
    #                      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, # morning, 06:00 to 11:00
    #                      0.4, 0.4, 0.4, 0.4, 0.1, 0.1, # afternoon, 12:00 to 17:00
    #                      0.1, 0.1, 0.4, 0.6, 0.6, 0.6 # evening, 18:00 to 23:00
    #                      ] * (number_days)
    
    # Keepign a constant flexibility of 15%
    flexibility_curve = [0.15] * 24 * number_days
    
    
    flexibility_curve = np.repeat(flexibility_curve, 60/minute_intervals)
    
    # Similarly, the willingness to shif the load is associated to WHEN the load will be shifted to. Thus, a symmetrical time window of X timestamps is defined
    # flexibility_window = [4, 4, 4, 4, 4, 4, # early morning, 00:00 to 05:00
    #                      2, 2, 2, 2, 2, 2, # morning, 06:00 to 11:00
    #                      2, 2, 2, 2, 2, 2, # afternoon, 12:00 to 17:00
    #                      2, 2, 3, 4, 4, 4 # evening, 18:00 to 23:00
    #                      ] * (number_days)
    
    flexibility_window = [2] * 24 * number_days
    
    flexibility_window = np.repeat(flexibility_window, 60/minute_intervals)

    # How much the load can be at minimum and maximum value considering the flexibility parameters?
    pairs, minload, maxload, maxload_future = min_max_loads(load_data, flexibility_curve, flexibility_window, plot=plot)

    if plot==True:
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot(flexibility_curve)
        ax[0].set_title('Flexibility curve')
        ax[0].set_ylabel('% of load')
        ax[1].plot(flexibility_window)
        ax[1].set_ylabel('Timestamps of flexibility')
        ax[1].set_xlabel('Timestamp')

    return flexibility_curve, flexibility_window, pairs, minload, maxload, maxload_future

# %% Automatic and controlled load shifting

def controlled_load_shift_sc(load_data, gen_data, pairs, flexibility_controlled, **kwargs):
    
    newload = load_data.copy()
    plot = kwargs.get('plot')
    minload = kwargs.get('minload')
    maxload = kwargs.get('maxload')
    
    # We first perform the controlled load shift from the microgrid control
    
    # we have a generation and demand forecasts and we can identify when the generation will be in excess - which enables the load shift
    idx_excess_gen = np.where(gen_data  > load_data)[0]
    
    # now we need to find which indexes have the indexes in idx_excess_gen as targets for load shifts. I.E. the timestamps in _pairs_ that contain the indexes
    # in idx_excess_gen as a component of their tuples. We'll shift the largest load
    for index in idx_excess_gen:
        idx_shiftable_loads = [pair[0] for pair in pairs if index in pair[1]]
        max_load_index = max(idx_shiftable_loads, key=lambda idex: newload[idex])
        for j in idx_shiftable_loads:
            if gen_data[index] > newload[index]:
                newload[index] = newload[index] + newload[j]*flexibility_controlled
                newload[j] = newload[j] - newload[j]*flexibility_controlled

    if plot == True:
        plt.figure()
        plt.plot(gen_data, linestyle='--', label='Generation')
        plt.plot(load_data, label='Original load')
        plt.plot(newload, label='New load with flexibility')
        plt.plot(minload, linestyle='--', linewidth=0.5, alpha=0.5, label='Minimum load')
        plt.plot(maxload, linestyle='--', linewidth=0.5, alpha=0.5, label='Maximum load')
        plt.legend()

    return newload


# %% EV behaviour

# Support function to get the closest index in an array
def index_time(target, time_intervals):
    abs_differences = np.abs(time_intervals - target)
    closest_index = np.argmin(abs_differences)
    return closest_index

def set_EV_behaviour(EV_parameters, number_days, final_time, minute_intervals, load_data, alpha=1.5, beta_= 7, 
                     std_dev = 2, mean_unplug = 8.5, min_unplug = 6, max_unplug = 11,
                     mean_plug = 17.5, min_plug = 12, max_plug = 22,
                     plot = True, random = False):
    """
    Setting the behaviour of the EV: how much EV charge is used per day, and the plug-in and plug-out times.

    Parameters
    ----------
    EV_parameters : Dict
        Dictionary setting the EV parameters.
    number_days : Int
        How many days the simulation is being run for.
    final_time : Int
        Final timestamp depending on the time intervals. 23 for 1-hour intervals, 23.75 for 15-minute intervals.
    minute_intervals : Int
        The time frequency of the array, 1-hour time intervals or 15-minute time intervals.
    load_data : Array
        Array with the load data.
    alpha : Float, optional
        Value to set the shape of the beta distribution on the left side. The default is 1.5.
    beta_ : Float, optional
        Value to set the shape of the beta distribution on the right side. The default is 7.
    std_dev : float, optional
        Standard deviation of the time which the EV is plugged-in/out. The default is 2.
    mean_unplug : Float, optional
        Mean time which the EV is unplugged. The default is 8.5.
    min_unplug : Float, optional
        The minimum hour which the EV is unplugged (morning). The default is 6.
    max_unplug : Float, optional
        The maximum hour that the EV is unplugged (morning). The default is 11.
    mean_plug : Float, optional
        Mean hour that the EV is plugged-in (afternoon/evening). The default is 17.5.
    min_plug : Float, optional
        The minimum hour that the EV is plugged-in. The default is 12.
    max_plug : Float, optional
        The maximum hour that the EV is plugged-in. The default is 22.
    plot : Bool, optional
        Whether we want to see the plots from the function. The default is True.
    random : Bool, optional
        Whether we want to set a random seed or use a set seed for tests. The default is False.

    Returns
    -------
    battery_used_percentages: Array
        Array with how much charge from the EV is used each day when driving.
    EV_plugged : Array
        Array with 1 for when the EV is plugged in, 0 for when the EV is out.

    """
    # Generating a probability density curve to get how much charge is depleted in the car every day
    x = np.linspace(0, 1, 100)
    pdf = beta.pdf(x, alpha, beta_)
    if plot == True:
        plt.plot(x * 100, pdf)
        plt.xlabel('Battery Usage (%)')
        plt.ylabel('Probability Density')
        plt.title('EV Battery Usage Probability Distribution')
        plt.grid(True)
        plt.show()

    if random == False:
    # Getting a random value and keeping it constant (seed = 42) for the tests
        np.random.seed(42)
    
    
    #### Setting how much EV charge is used:
    battery_usage_percentages = beta.rvs(alpha, beta_, size=number_days)

    #### Setting the plugged-in and out times:
    
    # EV is disconnected at 8:30 and gets reconnected at 17:00.
    # There is a fluctuation of 2h in the disconnection, and 1h in the connection
    
    # generating an array of times for the 15-minute intervals for us to get which are the indices needed
    time_intervals = np.linspace(0, final_time, int((24)*(60/minute_intervals)))

    # the closest index is our mean in the normal distribution for the unplugging time
    std_dev = 2
    mean_unplug = 8.5
    min_unplug = 6
    max_unplug = 11
    mean_plug = 17.5
    min_plug = 12
    max_plug = 22
    
    # generating the random distribution
    values_unplug = np.random.normal(index_time(mean_unplug, time_intervals), std_dev, number_days)
    values_unplug = [int(_) for _ in values_unplug]
    values_unplug = np.clip(values_unplug, index_time(min_unplug, time_intervals), index_time(max_unplug, time_intervals))
    
    values_plug = np.random.normal(index_time(mean_plug, time_intervals), std_dev, number_days)
    values_plug = [int(_) for _ in values_plug]
    values_plug = np.clip(values_plug, index_time(min_plug, time_intervals), index_time(max_plug, time_intervals))
    
    # Creating the EV plugged/unplugged pattern
    # We want to have variability in the times that the EV is plugged-in, thus we use the arrays defined above
    EV_plugged = np.ones_like(load_data)
    for i in range(len(EV_plugged)):
        if (i - values_unplug[i//int((24)*(60/minute_intervals))]) % int((24)*(60/minute_intervals)) <= (values_plug[i//int((24)*(60/minute_intervals))]
                                                                                                         - values_unplug[i//int((24)*(60/minute_intervals))]):
            EV_plugged[i] = 0
    
    if plot == True:
        plt.figure()        
        plt.plot(EV_plugged)
        plt.xlabel('Timestamp')
        plt.ylabel('EV plugged-in Yes/No')
        plt.title('EV plugged-in times')


    return battery_usage_percentages, EV_plugged










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
