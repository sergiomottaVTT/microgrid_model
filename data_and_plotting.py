# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:08:30 2023

@author: sdsergio
Script to work with data visualisation and generating plots

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


folder = r'figures/'


# %% Gathering ELSPOT price averages

# #elspot = pd.read_csv('data/Elspotprices.csv')

# price_data = pd.read_csv(r'data/price_data_5years.txt', delimiter='\t', header=None)
# price_data.columns = ['Timestamp', 'Price data']
# date_info = price_data['Timestamp'].to_list()
# date_noinfo = []
# for time in date_info:
#     date_noinfo.append(time.split('+')[0])
# # Creating a pandas Series from the list of strings
# s = pd.Series(date_noinfo)
# # Converting strings to datetime objects with timezone information
# localized_dt = pd.to_datetime(s)
# # Removing timezone information and formatting the datetime
# formatted_dt = localized_dt.dt.strftime('%Y-%m-%d %H:%M')
# index = pd.to_datetime(formatted_dt)
# price = price_data.copy()
# price.index = index
# price.drop(columns=['Timestamp'], inplace=True)

# price = price[price.index.year < 2023]
# price['Day'] = price.index.day
# price['Month'] = price.index.month
# price['Hour'] = price.index.hour
# # removing the leap day
# price = price[~((price['Day'] == 29) & (price['Month'] == 2))]
# # Calculating the average prices
# average_prices = price.groupby([price.index.month, price.index.day, price.index.hour])['Price data'].mean().reset_index()

# np.savetxt('data/price_average.txt', average_prices['Price data'].values)


# %% Plotting the synthetic load data

fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(load_data[7*4*24:2*7*4*24], linewidth=0.7, linestyle='-', color='k', label='Weekly profile')
axs[1].plot(load_data, linewidth=0.7, linestyle='-', color='k', label='Yearly profile')
# plt.plot(load_data_3[7*4*24:2*7*4*24], linewidth=0.7, linestyle=':', color='k', label='6MWh/year')
plt.title('Synthetic residential load data: Weekly profile')
axs[0].set_xlabel('Time step')
axs[0].set_title('Weekly load profile')
axs[1].set_xlabel('Time step')
axs[1].set_title('Yearly load profile')
axs[0].set_ylabel('Load (kWh)')
axs[1].set_ylabel('Load (kWh)')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(folder + 'fig1_synthetic_load_weekly_yearly.png', dpi=400)

#%% Plotting yearly data

fig, axs = plt.subplots(1, 1, sharey=True)
axs.plot(load_data, linewidth=0.7, linestyle='-', color='k', label='Yearly profile')
# plt.plot(load_data_3[7*4*24:2*7*4*24], linewidth=0.7, linestyle=':', color='k', label='6MWh/year')
plt.title('Synthetic residential load data: Yearly profile')
axs.set_xlabel('Time step')
axs.set_title('Yearly load profile')
axs.set_ylabel('Load (kWh)')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(folder + 'fig1_synthetic_load_yearly.png', dpi=400)

#%% Plotting weekly data

fig, axs = plt.subplots(1, 1, sharey=True)
axs.plot(load_data[7*4*24:2*7*4*24], linewidth=0.7, linestyle='-', color='k', label='Weekly profile')
# plt.plot(load_data_3[7*4*24:2*7*4*24], linewidth=0.7, linestyle=':', color='k', label='6MWh/year')
plt.title('Synthetic residential load data: Weekly profile')
axs.set_xlabel('Time step')
axs.set_title('Weekly load profile')
axs.set_ylabel('Load (kWh)')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(folder + 'fig1_synthetic_load_weekly.png', dpi=400)
#%% Comparison between load profiles
ax, fig = plt.subplots()
plt.plot(load_data[7*4*24:7*4*24+96], linewidth=0.7, linestyle='-', color='k', label='5MWh/year')
plt.plot(load_data_2[7*4*24:7*4*24+96], linewidth=0.7, linestyle='--', color='k', label='5.5MWh/year')
plt.plot(load_data_3[7*4*24:7*4*24+96], linewidth=0.7, linestyle=':', color='k', label='6MWh/year')
plt.title('Synthetic residential load data')
plt.xlabel('Time step')
plt.ylabel('Load (kWh)')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.savefig(folder + 'fig2_synthetic_load_comparison.png', dpi=400)

# %% PV generation
ax, fig = plt.subplots()
plt.plot(gen_data[7*4*24:2*7*4*24], linewidth=0.7, linestyle='-', color='k', label='PV generation')
plt.title('Synthetic solar power generation')
plt.xlabel('Time step')
plt.ylabel('Generation (kWh)')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.savefig(folder + 'fig3_synthetic_pv_gen.png', dpi=400)