# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:11:10 2023

@author: sdsergio

Functions for the microgrid simulation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm

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
            
       
        # removing the BESS charge/discharge from the difference
        difference = difference - BESS_charge + BESS_discharge

    # storing the values to be used in the microgrid dataframe
    BESS_SoC.append(BESS_parameters['SoC'])
    BESS_io.append(BESS_charge - BESS_discharge)

    return difference, BESS_parameters, BESS_SoC, BESS_io


# %% EV-RELATED FUNCTIONS

def  EV_behaviour(timestamp, difference, EV, price_data, price_threshold, minute_intervals):
    """
    Function to determine the EV behaviour, for each timestep of the simulation.

    Parameters
    ----------
    timestamp : int
        Simulation step.
    difference : float
        How much excess/demand is there for this timestep, after BESS charge/discharge.
    EV : EV Object
        Object from the EV class with all parameters for an EV.
    price_data : array
        Array with price information for each time step.
    price_threshold : float
        Threshold to define whe in is worth it to discharge BESS/EVs.
    minute_intervals : int
        The simulation time-step, e.g. 1-hour or 15-minutes.
    Returns
    -------
    difference: float
        How much energy is remaining after the EV operations.
    EV: EV Object
        Returning a modified EV object after the calculations in the function.
    """

    # If we have excess generation, the EV will will always try to charge 
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

    return difference, EV





# %% LOAD-SHIFTING FUNCTIONS 2

def load_shift_day_ahead(timestamp, difference, load_list, price_data, peak_limit, gen_shift=True, spot_price_following=True):
    """
    Function to perform the load shifting operations. When there is excess generation, we want to shift load towards the current time step, so we maximise
    self-consumption and reduce the amount of load imported from the grid. When there is no excess generation, we want to verify if we are consuming at
    lowest possible price, and thus check if we can shift load away from this timestamp to a cheaper time.
    
    We consider that all loads can be shifted back/forth, meaning that at each timestamp we will look into loads from the "past" and "future" to shift.
    Hence why "day_ahead": we are considering that these operations are performed with a bird's eye view of upcoming data, and full power to shift
    loads that haven't happened yet.
    
    Parameters
    ----------
    difference : float
        Amount of energy available in the microgrid. Difference between local generation, BESS_io, demand...
    timestamp : int
        Current time step of the simulation.
    load_list: list
        List of Load objects.
    price_data : array
        Array with the prices for each time step of the simulation.
    peak_limit: float
        The load limit at each timestamp, to avoid the creation of too large power peaks.
    gen_shift: Bool
        If we want to shift loads to maximise self-consumption and self-sufficiency. Default True.
    spot_price_following: Bool
        If we want to shift loads to minimise grid import costs, following the spot-price. Default True.

    Returns
    -------
    difference : float
        Amount of energy available in the microgrid after the load shifting operation.
    load_list: list
        List of Load objects, with their "newload" parameters modified after load shifting.
    """
    
    # When we have excess generation
    if difference >= 0 and gen_shift == True:
        # we want to perform a load shift TOWARDS this time to increase the self-consumption
        
        for load in load_list:
            if load.shifting == True:
                # Calculating the timestamp range (from flexibility curve) from which timestamps could be shifted
                idx_shiftable_loads = [pair[0] for pair in load.pairs if timestamp in pair[1]]
                
                # Since we are not doing the load shifting in real-time, but rather 
                # We do it time-based, not price-based. I.e. we are shifting the first load that we can from a timestamp perspective
                for i in range(len(idx_shiftable_loads)):
                    
                    if (difference > 0) and (load.newload[timestamp] <= peak_limit):
                        
                        # amount of load that will be shifted
                        load_shifted = load.newload[idx_shiftable_loads[i]] * load.flexibility_curve[idx_shiftable_loads[i]]
                
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
                        load.load_shift[timestamp].append(('gen/to', idx_shiftable_loads[i])) # TO (1) timestamp FROM idx_shiftable_loads[i]
                        load.load_shift[idx_shiftable_loads[i]].append(('gen/from', timestamp)) #FROM (-1) idx_shiftable_loads[i] TO timestamp
                        
                        # And the difference is now updated
                        difference = difference - load_shifted
                        #print('Updated difference:', difference)
         
                    
        # Seeing how the loads evolve in each timestep
        #plt.plot(load1.newload)

    # The load shifting is working as intended for improving self-consumption. Now we move to the case of the spot-price following
    
    ### TO-DO: Implement a limit to how much extra load can be added to each time-step!
    
    else: 
        
        for load in load_list:
            #print('Load being processed!---')
            # How much load can we shift? We can't shift more than the flexibility limit of the load
            if (load.newload[timestamp] >= load.load[timestamp] - (load.flexibility_curve[timestamp] * load.load[timestamp])) and (load.shifting == True)\
            and (spot_price_following == True):
                # The amount of load we can shift is then
                load_shifted = min(load.flexibility_curve[timestamp]*load.newload[timestamp], abs(difference))
                
                # We get which are the timestamps to where the load can be shifted
                times_to_shift = load.pairs[timestamp][1]
                
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
                    load.load_shift[timestamp].append(('price/to', times_to_shift[index_of_min_price])) # shifted FROM (-2) timestamp TO times_to_shift[index_of_min_price]
                    load.load_shift[times_to_shift[index_of_min_price]].append(('price/from', timestamp)) #shifted TO (2) times_to_shift[index_of_min_price] FROM timestamp
            
            
        # Improve this plot, this is a good one!
        #plt.plot(load1.newload, color='k', linewidth=0.8, alpha=0.7)


    #print('Secondary check: Difference at this timestamp is {:.2f}'.format(difference))

    return difference, load_list



def mg_day_ahead_operation(load_list, BESS_parameters, EV_list, gen_data, total_demand_after_shift, price_data, peak_limit, price_threshold, minute_intervals,
                           gen_shifting, spot_price_following):
    """
    Function to perform the microgrid operations in a "day-ahead" manner. That is, we are aware of the load, generation, and price well in advance
    and can control whether loads will be shifted forward or backwards in time (to an earlier or later time). 
    
    The load shift is performed in advance, then with the new information about the loads, the BESS and EVs are charged, and grid import/export is calculated.
    It is necessary to perform it like that because when we have loads shifting to earlier times, we are modifying a load value an an earlier timestamp,
    after we already ran through it in the for loop. Thus, we couldn't get consistent values for BESS, EV, and grid behaviours for each timestamp as the load
    values would change afterwards.
    
    There may be a better way to do this - probably there is - but for now this is how we operate the microgrid.
    1) perform the load shifting
    2) perform the BESS charging
    3) Perform the EV charging
    4) Calculate grid import/export
    
    Parameters
    ----------
    load_list : List
        List of Load objects.
    BESS_parameters : Dict
        Dictionary with the parameters of the BESS.
    EV_list : List
        List of EV objects.
    gen_data : Array
        Array of floats representing the generation available in the microgrid at each timestamp.
    total_demand_after_shift : Array
        Total demand, summing the consumption of all loads.
    price_data : Array
        Array with the spot-prices.
    peak_limit : Float
        Limit for peak powers.
    price_threshold : Float
        Value (in EUR) which is considered expensive enough to activate the BESS/EV discharging.
    minute_intervals : Int
        Simulation time-step, 15-minute or 60-minute intervals.

    Returns
    -------
    load_list : List
        List of Load objects, now modified with their "newload" parameter indicating the load shifts.
    total_demand_after_shift : Array
        Total demand of the microgrid loads, after the load shifting.
    BESS_SoC : List
        List of floats with the BESS SoC for each time-step
    BESS_io : List
        List of floats with the BESS charge/discharge for each time-step.
    EV_list : List
        List of EV objects with their parameters (SoC, charge/discharge) updated.
    grid_io : List
        List of floats with the grid import/export.
    """
    
    for timestamp in tqdm(range(len(gen_data)), desc='Day-ahead load shift'):
          
        ###### I: check excess generation  
        difference = gen_data[timestamp] - total_demand_after_shift[timestamp]
        
        ##### II: Perform the load shifting
        
        difference, load_list = load_shift_day_ahead(timestamp, difference, load_list, price_data, peak_limit,
                                                     gen_shift=gen_shifting, spot_price_following=spot_price_following)
        
        # Readjusting the total demand after the load shift
        total_demand_after_shift = np.sum([load.newload for load in load_list], axis=0)
            

    # After shifting the loads, we can also address the other assets in the microgrid
    # initialising simulation values
    grid_io, BESS_SoC, BESS_io = [], [], []
    
    for timestamp in tqdm(range(len(gen_data)), desc='Microgrid asset operation'):
          
        ###### I: check excess generation after the load was shifted
        difference = gen_data[timestamp] - total_demand_after_shift[timestamp]
       
        ##### II: Load shifting already performed
        
        ##### III: Setting the BESS behaviour
        difference, BESS_parameters, BESS_SoC, BESS_io = BESS_behaviour(difference, BESS_parameters, BESS_SoC, BESS_io, price_data, timestamp)
        
        
        ##### IV: Setting the EV behaviour   
      
        for EV in EV_list:
            difference, EV = EV_behaviour(timestamp, difference, EV, price_data, price_threshold, minute_intervals)
      
        
        # V: Setting the grid import/export behaviour
        grid_import, grid_export, grid_io = grid_behaviour(difference, grid_io)
        

    
    return load_list, total_demand_after_shift, BESS_SoC, BESS_io, EV_list, grid_io 






# %% CALCULATING AVAILABLE FLEXIBILITY (TO-DO)



# Flexibility availability calculation
# BESS_up, BESS_down, BESS_shift = [],[],[]
# gen_up, gen_down = [], []
# EV_up, EV_down, EV_shift = [],[],[]
# load_up, load_down = [], []
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







