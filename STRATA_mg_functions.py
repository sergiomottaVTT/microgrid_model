# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:11:10 2023

@author: sdsergio

Functions for the microgrid simulation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% BESS-RELATED FUNCTIONS:
    
def BESS_behaviour (difference, BESS_parameters, BESS_SoC, BESS_io, price_data, timestamp):
    """
    Function to determine the behaviour of the BESS in the simulation, for each time step.

    Parameters
    ----------
    difference : float
        THE DIFFERENCE BETWEEN THE LOCAL GENERATION AND THE DEMAND.
    BESS_parameters : dict
        DICTIONARY CONTAINING THE PARAMETERS OF THE BATTERY FOR THE TIMESTAMP.
    BESS_SoC : list
        LIST TO KEEP TRACK OF THE BESS STATE-OF-CHARGE DURING THE SIMULATION, APPENDED AT EACH TIMESTAMP.
    BESS_io : list
        LIST TO KEEP TRACK OF THE BESS CHARGE/DISCHARGE DURING THE SIMULATION, APPENDED AT EACH TIMESTAMP.
    price_data : array
        ARRAY WITH PRICE INFORMATION FOR EACH TIME STEP OF THE SIMULATION.
    timestamp : int
        INDICATION OF THE TIME STEP OF THE SIMULATION.

    Returns
    -------
    difference : float
        THE DIFFERENCE BETWEEN THE LOCAL GENERATION AND THE DEMAND AFTER THE BESS OPERATIONS
    BESS_parameters : dict
        DICTIONARY CONTAINING THE PARAMETERS OF THE BATTERY FOR THE TIMESTAMP UPDATED AFTER BESS OPERATIONS.
    BESS_SoC : list
        LIST TO KEEP TRACK OF THE BESS STATE-OF-CHARGE DURING THE SIMULATION, APPENDED AT EACH TIMESTAMP.
    BESS_io : list
        LIST TO KEEP TRACK OF THE BESS CHARGE/DISCHARGE DURING THE SIMULATION, APPENDED AT EACH TIMESTAMP.

    """

    # Do we want the BESS to be charged by the grid, or only by the excess generation?
    
    # If we have excess generation, the BESS will always try to charge    
    if difference > 0:
        BESS_discharge = 0
        
        BESS_charge = min(difference, BESS_parameters['cRate'], BESS_parameters['capacity'] - BESS_parameters['SoC'])
        BESS_parameters['SoC'] = min(BESS_parameters['SoC'] + BESS_charge, BESS_parameters['capacity'])
        
        # updating the available difference in generation after the BESS is charged
        difference = difference - BESS_charge

    # If we don't have excess generation, the BESS will discharge if it's control parameter is set for it
    else:
        # define if the BESS will charge or discharge
        # DISCHARGE
        if (BESS_parameters['Control'] == 'load_threshold') and (abs(difference) >= BESS_parameters['Control setpoint']):
            BESS_discharge = min(BESS_parameters['cRate'], abs(difference), BESS_parameters['SoC'])
            BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
            BESS_charge = 0
        elif (BESS_parameters['Control'] == 'price_threshold') and (price_data[timestamp] >= BESS_parameters['Control setpoint']):
            BESS_discharge = min(BESS_parameters['cRate'], abs(difference), BESS_parameters['SoC'])
            BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
            BESS_charge = 0
        # CHARGE
        elif (BESS_parameters['Grid enabled'] == True) and (BESS_parameters['SoC'] < BESS_parameters['SoC threshold']*BESS_parameters['capacity']):
            BESS_charge = min(BESS_parameters['cRate'], BESS_parameters['capacity'] - BESS_parameters['SoC'])
            BESS_parameters['SoC'] = min(BESS_parameters['SoC'] + BESS_charge, BESS_parameters['capacity'])
            BESS_discharge = 0
        # NOTHING HAPPENS - BESS idle
        else:
            BESS_charge = 0
            BESS_discharge = 0
            BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
            
            
        # if (BESS_parameters['Grid enabled'] == True) and (BESS_parameters['SoC'] < BESS_parameters['SoC threshold']*BESS_parameters['capacity']):
        #     if (BESS_parameters['Control'] == 'load_threshold') and (abs(difference) < BESS_parameters['Control setpoint']):
        #     # We charge the BESS instead of discharging it
        #         BESS_charge = min(BESS_parameters['cRate'], BESS_parameters['capacity'] - BESS_parameters['SoC'])
        #         BESS_parameters['SoC'] = min(BESS_parameters['SoC'] + BESS_charge, BESS_parameters['capacity'])
        #         BESS_discharge = 0
        #     elif (BESS_parameters['Control'] == 'price_threshold') and (price_data[timestamp] < BESS_parameters['Control setpoint']):
        #         BESS_charge = min(BESS_parameters['cRate'], BESS_parameters['capacity'] - BESS_parameters['SoC'])
        #         BESS_parameters['SoC'] = min(BESS_parameters['SoC'] + BESS_charge, BESS_parameters['capacity'])
        #         BESS_discharge = 0
        #     else:
        #         BESS_charge = 0
        #         BESS_discharge = 0
        #         BESS_parameters['SoC'] = BESS_parameters['SoC']
            
        # else:
        #     BESS_charge = 0
        #     if BESS_parameters['Control'] == 'load_threshold':
        #         # the BESS discharges only if the load is high enough
        #         if abs(difference) >= BESS_parameters['Control setpoint']:
        #             BESS_discharge = min(BESS_parameters['cRate'], abs(difference), BESS_parameters['SoC'])
        #             BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
        #         else:
        #             BESS_discharge = 0
        #             BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
            
        #     elif BESS_parameters['Control'] == 'price_threshold':
        #         # The BESS discharges only if the price is high enough
        #         if price_data[timestamp] > BESS_parameters['Control setpoint']:
        #             BESS_discharge = min(BESS_parameters['cRate'], abs(difference), BESS_parameters['SoC'])
        #             BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
        #         else:
        #             BESS_discharge = 0
        #             BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
            
        #     else: # no control defined
        #         print('Warning: no BESS control defined!')
        #         BESS_discharge = 0
        #         BESS_parameters['SoC'] = max(BESS_parameters['SoC'] - BESS_discharge, 0)
        
        # removing the BESS charge/discharge from the difference
        difference = difference - BESS_charge + BESS_discharge

    # storing the values to be used in the microgrid dataframe
    BESS_SoC.append(BESS_parameters['SoC'])
    BESS_io.append(BESS_charge - BESS_discharge)

    return difference, BESS_parameters, BESS_SoC, BESS_io


# %% EV-RELATED FUNCTIONS

def  EV_behaviour(timestamp, difference, EV_plugged, EV_parameters, price_data, price_threshold, 
                  battery_usage_percentages, minute_intervals, EV_out_for_the_day, EV_SoC, EV_io):
    """
    Function to determine the EV behaviour, for each timestep of the simulation.

    Parameters
    ----------
    timestamp : int
        Simulation step.
    difference : float
        How much excess/demand is there for this timestep, after BESS charge/discharge.
    EV_plugged : array
        Array with 1s for when the EV is plugged-in, 0s for when the EV is out.
    EV_parameters : dict
        Dictionary with the EV parameters, including SoC.
    price_data : array
        Array with price information for each time step.
    price_threshold : float
        Threshold to define whe in is worth it to discharge BESS/EVs.
    battery_usage_percentages : array
        Array with the daily EV charge use.
    minute_intervals : int
        The simulation time-step, e.g. 1-hour or 15-minutes.
    EV_out_for_the_day : int
        Flag variable to define when the EV is unplugged and discharged during the ride.
    EV_SoC : list
        List to register the SoC of the EV for each simulation step.
    EV_io : list
        List to register the charge/discharge of the EV for each simulation step.

    Returns
    -------
    difference: float
        How much energy is remaining after the EV operations.
    EV_parameters: dict
        Dictionary with the EV parameters, with updated SoC.
    EV_SoC : list
        List to register the SoC of the EV for each simulation step.
    EV_io : list
        List to register the charge/discharge of the EV for each simulation step.
    EV_out_for_the_day : int
        Flag variable to define when the EV is unplugged and discharged during the ride.
    """
    
    # If we have excess generation, the EV will will always try to charge    

    if EV_plugged[timestamp] == 1:
        # the EV is available to be charged
        
        if (difference > 0):
            # we charge it with the excess generation
            EV_charge = min(difference, EV_parameters['cRate'], EV_parameters['capacity'] - EV_parameters['SoC'])
            EV_parameters['SoC'] = min(EV_parameters['SoC'] + EV_charge, EV_parameters['capacity'])
            EV_discharge = 0
            
        else:
            # should we charge or discharge (v2g)?
            if (EV_parameters['V2G'] == True) and (EV_parameters['SoC'] > 
                EV_parameters['discharge threshold'][0]*EV_parameters['capacity']) and (price_data[timestamp] > price_threshold):
                # It is worthwhile to discharge the EV to avoid paying a high price to grid imports
                
                EV_discharge = min(EV_parameters['cRate'], abs(difference), (EV_parameters['SoC'] 
                                   - EV_parameters['discharge threshold'][1]*EV_parameters['capacity']))
                EV_parameters['SoC'] = max(EV_parameters['SoC'] - EV_discharge, 0)
                EV_charge = 0
                
            else:
                # It's better to charge the EV or don't do anything with it
    
                # we charge it by importing from the grid
                EV_charge = min(EV_parameters['cRate'], EV_parameters['capacity'] - EV_parameters['SoC'])
                EV_parameters['SoC'] = min(EV_parameters['SoC'] + EV_charge, EV_parameters['capacity'])
                EV_discharge = 0
        
        difference = difference - EV_charge + EV_discharge
        EV_out_for_the_day = 0
        
    else:
        if EV_out_for_the_day == 0:
            EV_parameters['SoC'] = EV_parameters['SoC'] - battery_usage_percentages[timestamp 
                                                                                    // int((24)*(60/minute_intervals))]*EV_parameters['SoC']
            EV_out_for_the_day = 1
        
        EV_charge = 0
        EV_discharge = 0
    
       
    EV_io.append(EV_charge - EV_discharge) 
    EV_SoC.append(EV_parameters['SoC'])
    
    return difference, EV_parameters, EV_SoC, EV_io, EV_out_for_the_day


# %% lOAD-SHIFTING FUNCTIONS:


def load_shift_behaviour (difference, newload, load_data, timestamp, pairs, load_shift, flexibility_curve, price_data):
    """
    Function to perform the load shifting operations. When there is excess generation, we want to shift load towards the current time step, so we maximise
    self-consumption and reduce the amount of load imported from the grid. When there is no excess generation, we want to verify if we are consuming at
    lowest possible price, and thus check if we can shift load away from this timestamp to a cheaper time.

    Parameters
    ----------
    difference : float
        Amount of energy available in the microgrid. Difference between local generation, BESS_io, demand...
    newload : array
        Array of floats that represent the new, changed demand of the microgrid accounting for its flexibility.
    timestamp : int
        Current time step of the simulation.
    pairs : list
        List of tuples that associate the current time step with the possible time steps for flexibility.
        Calculated with flexibility_curve.
    load_shift : array
        Array that indicates whether there was a load shift at the current time step. Used for tracking purposes.
    flexibility_curve : array
        Flexibility curve that gives an interval of how many time steps the load can be shifted to/from.
    price_data : array
        Array with the prices for each time step of the simulation.

    Returns
    -------
    difference : float
        Amount of energy available in the microgrid after the load shifting operation.
    newload : array
        Array of the demand after the load shifting operation. The demand in the current time step may have been modified.

    """
    
    spot_price_following = True
    
    # There is excess generation -- can we shift some later load to be used now?
    # i.e., we are shifting load TOWARDS this timestamp
    if difference >= 0:
                     
        # Calculating the timestamp range (from flexibility curve) from which timestamps could be shifted
        idx_shiftable_loads = [pair[0] for pair in pairs if timestamp in pair[1]]
        # However, we can't alter the past, so we can only get indexes which are larget than timestamp
        idx_shiftable_loads = [element for element in idx_shiftable_loads if element > timestamp]
        
        # Performing the shifts from upcoming timestamps
        for i in range(len(idx_shiftable_loads)):
    
            if (difference > 0) and (newload[timestamp] < np.max(newload)) and \
                (load_shift[idx_shiftable_loads[i]] >= 0):
                # How much load can be shifted from this first timestamp [i]
                load_shifted = newload[idx_shiftable_loads[i]]*flexibility_curve[idx_shiftable_loads[i]]
                
                # we don't want to create any new peaks in the consumption
                if (newload[timestamp] + load_shifted) > np.max(newload):
                    excess = newload[timestamp] + load_shifted - np.max(newload)
                    load_shifted = load_shifted - excess
                    
                newload[timestamp] = newload[timestamp] + load_shifted
                newload[idx_shiftable_loads[i]] = newload[idx_shiftable_loads[i]] - load_shifted
                # marking that there was a load shift executed here and timestamp has extra load
                load_shift[timestamp] = load_shift[timestamp] + 1 # shift TO here
                load_shift[idx_shiftable_loads[i]] = load_shift[idx_shiftable_loads[i]] - 1 # shift FROM here
                # removing this extra load shifted from the difference (positive)
                difference = difference - load_shifted
        
    # There is no excess generation -- can we shift some load to a cheaper time? 
    # i.e., we are shifting load AWAY from this timestamp
    else: 
               
        # How much load can we shift? We can't shift more than the flexibility limit of the load
        load_shifted = min(flexibility_curve[timestamp]*newload[timestamp], abs(difference))
        # we check if this load can and should be shifted somehow to another time
        if (load_shift[timestamp] >= 0) and (spot_price_following == True):   # we can only shift load AWAY it if it wasn't shifted AWAY before
            # We get which are the timestamps to where the load can be shifted
            # However, we can't change the past loads, so we can only shift the current load to a future timestamp
            times_to_shift = [item for item in pairs[timestamp][1] if item > timestamp]
            # what is the current price
            current_price = price_data[timestamp]
            # We want to get the prices of all these timestamps to see when it is the lowest price in this interval
            prices_window = price_data[times_to_shift]
            
            # are there any times in the shifting window that would have smaller prices?
            smaller_prices = [price for price in prices_window if price < current_price]
            if len(smaller_prices) >= 1:
                # if there are smaller prices in the window, where is the minimum?
                index_of_min_price = np.where(prices_window == (min(smaller_prices)))[0][0]

                # we don't want to create any new peaks in the consumption
                if (newload[times_to_shift[index_of_min_price]] + load_shifted) > np.max(newload):
                    excess = newload[times_to_shift[index_of_min_price]] + load_shifted - np.max(newload)
                    load_shifted = load_shifted - excess
                
                # then we perform the load shift
                newload[times_to_shift[index_of_min_price]] = newload[times_to_shift[index_of_min_price]] +\
                    load_shifted
                
                newload[timestamp] = newload[timestamp] - load_shifted
                
                # and let's mark that the load was shifted (can't be shifted again), and the remaining load at timestamp is smaller (difference is negative)
                difference = difference + load_shifted 
                load_shift[timestamp] = load_shift[timestamp] - 1 # shifted FROM here
                load_shift[times_to_shift[index_of_min_price]] = load_shift[times_to_shift[index_of_min_price]] + 1 #shifted TO here



    return difference, newload, load_shift


# %% GRID-RELATED FUNCTIONS:


def grid_behaviour (difference, grid_io):
    """
    Function to calculate the behaviour of the grid and the import/export

    Parameters
    ----------
    difference : float
        Float that indicates how much energy is still available in the microgrid.
    grid_io : list
        List to keep track of the import/export; updated and appended in each time step.

    Returns
    -------
    grid_import : float
        Float to indicate how much is imported from the grid in the current time step.
    grid_export: float
        Float to indicate how much is exported to the grid in the current time step.
    grid_io : list
        Updated grid_io list with the import/export for the current time step.

    """
    


# TODO: implement the grid availability and VoLL calculations

    # there is excess energy still, and we will export to the grid
    if difference > 0:
        grid_export = difference # export is positive since difference = generation - demand - BESS_io
        grid_import = 0
    
    # since we can't meet the load with the generation, we import from the grid what is needed   
    else:          
        grid_import = min(difference, 0) # import should be negative
        grid_export = 0

    # combining grid import and export (+ is export, - is import)
    grid_io.append(grid_import + grid_export)
    
    return grid_import, grid_export, grid_io    







