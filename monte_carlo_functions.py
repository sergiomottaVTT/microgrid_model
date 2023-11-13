<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:16:38 2023

@author: sdsergio
"""

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









=======
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:16:38 2023

@author: sdsergio
"""

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









>>>>>>> origin/main
