# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:07:59 2023

@author: sdsergio
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta


# %% 

class Load:
    """
    A class to define Load objects.
    This class represents individual loads within a microgrid. It can be used to represent an aggregated load
    i.e., a house, or an individual load (such as a heat pump).
    
    Attributes:
        load: original load time-series
        newload: time-series representing the load behaviour after load shifting is implemented
        load_shift: array containing lists of tuples that indicate when loads have been shifted, from which timestep, and to which timestep.
        flexibility_curve: array indicating how much flexibility (%) of the load at each time interval is available to be shifted
        flexibility_interval: array indicating how many time-steps the load can be shifted back/forth.
        shifting: bool attribute to specify whether the load is flexible or not, and if shifting can be implemented.
    """
    
    def __init__(self, load, newload, shifting):
        self.load = copy.deepcopy(load)
        self.newload = copy.deepcopy(newload)
        self.load_shift = [[] for _ in range(len(load))]
        self.shifting = shifting

    ### we want to be able to modify the attibute without modifying the original variable, hence why we implement
    ### "copy.deepcopy" when initialising our load object.
    
    
    def min_max_loads(load_data, flexibility_curve, flexibility_window, **kwargs):
        """
        Method to calculate the minimum possible load, maximum possible load, and most importantly,
        to define which loads can be shifted to which timesteps (pairs).

        Parameters
        ----------
        load_data : Array
            Array with the time-series showing the load behaviour.
        flexibility_curve : Array
            Array with the percentage of load that can be shifted at each timestep.
        flexibility_window : Array
            Array with the time interval (number of timesteps back/forth) that the load can be shifted to.
        **kwargs :
            plot: Bool, wether we want to plot or not the results.

        Returns
        -------
        pairs : List
            List of tuples that indicate to which timesteps a load at timestep X can be shifted to.
        minload : Array
            Minimum load possible at each timestep after load shift.
        maxload : Array
            Maximum load possible at each timestep after load shift.
        maxload_future : Array
            Maximum load possible if we consider only future load shifting (no changing the past, i.e. real-time operation).

        """
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
    
    
    def define_flexibility(self, number_days, minute_intervals, plot=False):
        """
        Method to define the flexibility associated to a Load object.

        Parameters
        ----------
        number_days : int
            How long is the simulation.
        minute_intervals : int
            Simulation time-step: 15-minute or 60-minute intervals.
        plot : Bool, optional
            To define whether we want to visualise the output of the curves. The default is False.

        Returns
        -------
        flexibility_curve : Array
            Array with the percentage of load that can be shifted at each timestep.
        flexibility_window : Array
            Array with the time interval (number of timesteps back/forth) that the load can be shifted to.
        minload : Array
            Minimum load possible at each timestep after load shift.
        maxload : Array
            Maximum load possible at each timestep after load shift.
        maxload_future : Array
            Maximum load possible if we consider only future load shifting (no changing the past, i.e. real-time operation).

        """
        # Each timestamp has a different amount of flexibility, based on people's willingness to modify their consumption behaviour (% of the load)
        # flexibility_curve = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, # early morning, 00:00 to 05:00
        #                      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, # morning, 06:00 to 11:00
        #                      0.4, 0.4, 0.4, 0.4, 0.1, 0.1, # afternoon, 12:00 to 17:00
        #                      0.1, 0.1, 0.4, 0.6, 0.6, 0.6 # evening, 18:00 to 23:00
        #                      ] * (number_days)
        
        # Keepign a constant flexibility of 15%
        flexibility_curve = [0.15] * 24 * number_days
        
        
        self.flexibility_curve = np.repeat(flexibility_curve, 60/minute_intervals)
        
        # Similarly, the willingness to shif the load is associated to WHEN the load will be shifted to. Thus, a symmetrical time window of X timestamps is defined
        # flexibility_window = [4, 4, 4, 4, 4, 4, # early morning, 00:00 to 05:00
        #                      2, 2, 2, 2, 2, 2, # morning, 06:00 to 11:00
        #                      2, 2, 2, 2, 2, 2, # afternoon, 12:00 to 17:00
        #                      2, 2, 3, 4, 4, 4 # evening, 18:00 to 23:00
        #                      ] * (number_days)
        
        flexibility_window = [2] * 24 * number_days
        
        self.flexibility_window = np.repeat(flexibility_window, 60/minute_intervals)
    
        # How much the load can be at minimum and maximum value considering the flexibility parameters?
        self.pairs, self.minload, self.maxload, self.maxload_future = Load.min_max_loads(self.load, flexibility_curve, flexibility_window, plot=plot)
    
        if plot==True:
            fig, ax = plt.subplots(2,1, sharex=True)
            ax[0].plot(flexibility_curve)
            ax[0].set_title('Flexibility curve')
            ax[0].set_ylabel('% of load')
            ax[1].plot(flexibility_window)
            ax[1].set_ylabel('Timestamps of flexibility')
            ax[1].set_xlabel('Timestamp')
    
        return self.flexibility_curve, self.flexibility_window, self.pairs, self.minload, self.maxload, self.maxload_future


# %%

class EV:
    """
    A class to represent EVs in the microgrid.
    
    Attributes:
        capacity: float representing the EV battery capacity
        cRate: float charging-rate the EV supports. Limited by the charging station, should be the same for all EVs.
        SoC: float indicating the initial SoC of the EV at the start of the simulation.
        discharge_threshold: Tuple with floats indicating (SoC which allows the EV to be discharged, minimum SoC acceptable to discharge to).
        plugged_array: Array indicating when the EV is plugged-in (1 or 0) in the microgrid.
        battery_use: Array of floats indicating how much battery % is used per day when the EV is disconnected and driven.
        V2G: bool to indicate whether the EV has V2G capabilities
        day_disconnect: 1/0 flag to indicate if the EV was driven in the day.
        EV_SoC: list to keep track of the SoC during the simulation.
        EV_io: list to keep track of the charge/discharge during the simulation.
    """
    
    def __init__(self, capacity, cRate, SoC, discharge_threshold, V2G=False):
        self.capacity = capacity
        self.cRate = cRate
        self.SoC = copy.deepcopy(SoC)
        self.discharge_threshold = discharge_threshold
        # self.plugged_array = copy.deepcopy(EV_plugged)
        # self.battery_use = battery_use
        self.V2G = V2G
        self.day_disconnect = 0
        self.EV_SoC = []
        self.EV_io = []
    
    
    
    # Support function to get the closest index in an array
    def index_time(target, time_intervals):
        abs_differences = np.abs(time_intervals - target)
        closest_index = np.argmin(abs_differences)
        return closest_index

    def set_EV_behaviour(self, number_days, final_time, minute_intervals, load_data, alpha=1.5, beta_= 7, 
                         std_dev = 2, mean_unplug = 8.5, min_unplug = 6, max_unplug = 11,
                         mean_plug = 17.5, min_plug = 12, max_plug = 22,
                         plot = True, random = False):
        """
        Setting the behaviour of the EV: how much EV charge is used per day, and the plug-in and plug-out times.
    
        Parameters
        ----------
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
        # Beta PDF was used as initial test
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
        self.battery_use = beta.rvs(alpha, beta_, size=number_days)
    
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
        values_unplug = np.random.normal(EV.index_time(mean_unplug, time_intervals), std_dev, number_days)
        values_unplug = [int(_) for _ in values_unplug]
        values_unplug = np.clip(values_unplug, EV.index_time(min_unplug, time_intervals), EV.index_time(max_unplug, time_intervals))
        
        values_plug = np.random.normal(EV.index_time(mean_plug, time_intervals), std_dev, number_days)
        values_plug = [int(_) for _ in values_plug]
        values_plug = np.clip(values_plug, EV.index_time(min_plug, time_intervals), EV.index_time(max_plug, time_intervals))
        
        # Creating the EV plugged/unplugged pattern
        # We want to have variability in the times that the EV is plugged-in, thus we use the arrays defined above
        self.plugged_array = np.ones_like(load_data)
        for i in range(len(self.plugged_array)):
            if (i - values_unplug[i//int((24)*(60/minute_intervals))]) % int((24)*(60/minute_intervals)) <= (values_plug[i//int((24)*(60/minute_intervals))]
                                                                                                             - values_unplug[i//int((24)*(60/minute_intervals))]):
                self.plugged_array[i] = 0
        
        if plot == True:
            plt.figure()        
            plt.plot(self.plugged_array)
            plt.xlabel('Timestamp')
            plt.ylabel('EV plugged-in Yes/No')
            plt.title('EV plugged-in times')
    
    
        return self.battery_use, self.plugged_array






#