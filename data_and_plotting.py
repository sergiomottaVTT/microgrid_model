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

#%% Load flexibility and load flexibility window

from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Arc

minload = np.zeros_like(load1.minload[7*4*24:7*4*24+96])
minload[20:46] = load1.minload[7*4*24+20:7*4*24+46]

ax, fig = plt.subplots()
plt.plot(load_data[7*4*24:7*4*24+96], linewidth=0.7, linestyle='-', color='k', label='5MWh/year')
plt.scatter(70, load_data[7*4*24+70] - 0.15*load_data[7*4*24+70], linewidth=0.7, marker='o', color='k', label='Load flexibility at t = H')
#plt.fill_between([75-10, 75+10], min(load_data[7*4*24:7*4*24+96]), max(load_data[7*4*24:7*4*24+96]), color='gray', alpha=0.3)
plt.axvline(x=70-10, color='gray', linestyle='--')
plt.axvline(x=70+10, color='gray', linestyle='--')
plt.plot([70, 70], [load_data[7*4*24+70] - 0.15*load_data[7*4*24+70], load_data[7*4*24+70]], color='gray', linestyle='--')

plt.annotate('Load(H) - Load flexibility $L_f(H)$', xy=(70, load_data[7*4*24+70] - 0.15*load_data[7*4*24+70]), xytext=(5, load_data[7*4*24+70] - 0.15*load_data[7*4*24+70]),
             arrowprops=dict(facecolor='black', arrowstyle='->'))


plt.annotate('Flexibility window $T_f(H)$', xy=(70, 0.5), xytext=(55, 0.5),
             )



# con = ConnectionPatch(xyA=(70-10, 0.1), xyB=(70+10, 0.1), coordsA='data', coordsB='data',
#                       axesA=plt.gca(), axesB=plt.gca(), linestyle="dashed", color="black")
# plt.gca().add_patch(con)


plt.title('Synthetic residential load data')
plt.xlabel('Time step')
plt.ylabel('Load (kWh)')
#plt.legend()
plt.xticks(ticks=[70], labels=['H'], fontstyle='italic')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.savefig(folder + 'fig3_minload.png', dpi=400)





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



# %% Spot price data


ax, fig = plt.subplots()
plt.plot(price_data, linewidth=0.7, linestyle='-', color='k', label='Average hourly spot prices')
plt.title('Average hourly spot price data 2018-2022')
plt.xlabel('Time step')
plt.ylabel('EUR/kWh')
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.savefig(folder + 'fig4_avg_spot_price.png', dpi=400)





# %% Energy Community Behaviour

microgrid_simulation = pd.read_pickle(r'data/results/0_25flex_1h_20houses_full.pkl')


import matplotlib.ticker as ticker
import matplotlib.dates as mdates

columns = ['Total demand', 'Total demand_shift', 'Generation']
dates = ['2022-03-15','2022-03-17']

mg_plot = microgrid_simulation.loc[dates[0], columns]

ax, fig = plt.subplots()
plt.plot(mg_plot['Total demand'], linewidth=0.7, linestyle='-', color='k', label='Original demand')
plt.plot(mg_plot['Total demand_shift'], linewidth=0.9, linestyle='--', color='k', label='Demand after load shift')
plt.plot(mg_plot['Generation'], linewidth=0.7, alpha=0.5, linestyle=':', color='k', label='Local PV generation')
plt.title('Load shift to improve self-consumption and self-sufficiency \n and reduce costs with grid imports')
plt.xlabel('Time')
plt.ylabel('kWh')
plt.xticks(rotation=45)
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.savefig(folder + 'fig8_loadshift_3h.png', dpi=400)


# %% Energy Community Behaviour

microgrid_simulation = pd.read_pickle(r'data/results/01flex_1h_20houses_spot.pkl')

import matplotlib.ticker as ticker
import matplotlib.dates as mdates

columns = ['Total demand', 'Total demand_shift', 'Price data']
dates = ['2022-03-15','2022-03-17']

mg_plot = microgrid_simulation.loc[dates[0], columns]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(mg_plot['Total demand'], linewidth=0.7, linestyle='-', color='k', label='Original demand')
ax[0].plot(mg_plot['Total demand_shift'], linewidth=0.9, linestyle='--', color='k', label='Demand after load shift')
#ax[0].plot(mg_plot['Generation'], linewidth=0.7, alpha=0.5, linestyle=':', color='k', label='Local PV generation')
ax[0].set_title('Load shift to maximise self-consumption and self-sufficiency')
ax[0].set_ylabel('kWh')
ax[0].legend()
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax[1].plot(mg_plot['Price data'], linewidth=0.7, linestyle='-', color='k', label='Spot price')
#ax[0].plot(mg_plot['Generation'], linewidth=0.7, alpha=0.5, linestyle=':', color='k', label='Local PV generation')
#ax[1].set_title('Load shift to maximise self-consumption and self-sufficiency')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('EUR/kWh')
#ax[1].xticks(rotation=45)
ax[1].legend()
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.savefig(folder + 'fig6_spotprice_loadshift.png', dpi=400)






















