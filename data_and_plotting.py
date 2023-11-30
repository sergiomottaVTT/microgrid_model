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

elspot = pd.read_csv('data/Elspotprices.csv')




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