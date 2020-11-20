#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analysis on Unmet Demand

Created on Fri Nov 13 12:55:20 2020

@author: xuanlin
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
trip_file = '2018Q12_top25station_trips.csv'
sensor_file = '2018Q12_top25station_sensor.csv'
station_info_file = 'station_info.csv'


top_station_ids = [35, 192,  91,  77,  43, 133, 174,  81,  76,  90, 177, 287, 268,
                   195,  85, 283, 100,  66, 110,  52, 181,  48,  59, 176,  49]


# Load Data
trips = pd.read_csv(path + trip_file)
print("Trip file size: {}".format(trips.shape))
sensor = pd.read_csv(path + sensor_file)
print("Sensor file size: {}".format(sensor.shape)) # (640100, 18)
info = pd.read_csv(path + station_info_file)
print("Station Info file size: {}".format(info.shape))

# Severity evaluation
# Only consider bike outage during the operating hours
bike_outage = sensor[(sensor['available_bikes'] == 0) & (sensor['hour'] >= 7) & (sensor['hour'] <= 20)] \
    .groupby(['id', 'date'])['date_clean'].count().reset_index() # (1486, 2)
bike_outage.columns = ['id', 'date', 'period']

dock_outage = sensor[(sensor['available_docks'] == 0) & (sensor['hour'] >= 7) & (sensor['hour'] <= 20)] \
    .groupby(['id', 'date'])['date_clean'].count().reset_index()  # (892, 2)
dock_outage.columns = ['id', 'date', 'period']

total_station_day = 25 * 181
print("Bike outage ratio: {}".format(bike_outage.shape[0] / total_station_day)) # 0.39, measured in station_day
print("Dock outage ratio: {}".format(dock_outage.shape[0] / total_station_day)) # 0.28


# Example of bike and dock outages
top_bike_outage = bike_outage.sort_values('period', ascending = False).head(50)
top_dock_outage = dock_outage.sort_values('period', ascending = False).head(50)

# Inflows and Outflows
keys = ['station_id', 'date', 'time_bin']
window_duration = 10 #min
out_df = trips[trips['from_station_id'].isin(top_station_ids)].copy()
out_df['start_time'] = pd.to_datetime(out_df['start_time'])
out_df['start_date'] = out_df['start_time'].dt.date
out_df['start_time_bin'] = pd.cut(out_df['start_time'].dt.hour * 60 + out_df['start_time'].dt.minute,
                                 bins = range(0, 1441, window_duration),
                                 include_lowest = True,
                                 labels = range(0, 1431, window_duration))  #.astype('category').cat.codes * 2
out_df = out_df.groupby(['from_station_id', 'start_date', 'start_time_bin'])['trip_id'].count().reset_index()
out_df.columns =keys + ['out_trips']
print('Outflow DF shape: {}'.format(out_df.shape)) # Outflow DF shape: (651600, 4)

in_df = trips[trips['to_station_id'].isin(top_station_ids)].copy()
in_df['end_time'] = pd.to_datetime(in_df['end_time'])
in_df['end_date'] = in_df['end_time'].dt.date
in_df['end_time_bin'] = pd.cut(in_df['end_time'].dt.hour * 60 + in_df['end_time'].dt.minute,
                                 bins = range(0, 1441, window_duration),
                                 include_lowest = True,
                                 labels = range(0, 1431, window_duration))  #.astype('category').cat.codes * 2
in_df = in_df.groupby(['to_station_id', 'end_date', 'end_time_bin'])['trip_id'].count().reset_index()
in_df.columns =keys + ['in_trips']
print('Inflow DF shape: {}'.format(in_df.shape)) # Outflow DF shape: (651600, 4)



def plot_outage(station_id, date, data = sensor, out_trip_df = out_df, in_trip_df = in_df):
    # data processing
    to_plot = data[(data['id'] == station_id) & (data['date_clean'].str.slice(0,10) == date)].copy()
    N = to_plot.shape[0]
    outflow = out_trip_df[(out_trip_df['station_id'] == station_id) & 
                          (pd.to_datetime(out_trip_df['date']) == date)].copy()
    inflow = in_trip_df[(in_trip_df['station_id'] == station_id) & 
                          (pd.to_datetime(in_trip_df['date']) == date)].copy()
    
    # plot
    plt.figure(figsize=(15,4))
    plt.plot(range(N), to_plot['available_bikes'])
    plt.plot(range(N), to_plot['docks_in_service'], linestyle = 'dashed', color = 'red')
    plt.plot(range(outflow.shape[0]), outflow['out_trips'].cumsum())
    plt.plot(range(inflow.shape[0]), inflow['in_trips'].cumsum())
    plt.xticks(range(0, N, 2), to_plot['date_clean'].str.slice(10, 16)[::2], rotation = 90)
    plt.title('Available Bikes: {}, {}'.format(station_id, date))
    plt.legend(["# bikes in station", 'capacity', 'outflow', 'inflow'])
    plt.show()

# Plot the top outage cases
plot_outage(top_dock_outage.iloc[35, 0], str(top_dock_outage.iloc[35, 1]))
plot_outage(top_bike_outage.iloc[35, 0], str(top_bike_outage.iloc[35, 1]))






"""
# Fix the AM PM issue...Need to implement for all data
a = list(map(lambda x, y: pd.to_datetime(x) + pd.DateOffset(hours=12) if y == 'PM' else x, 
         sensor['date_clean'], 
         sensor['timestamp'].str.slice(-2,)))
sensor['date_clean'] = list(map(str, a))
                                           

sensor['hour'] = pd.to_datetime(sensor['date_clean']).dt.hour
sensor['date'] = pd.to_datetime(sensor['date_clean']).dt.date
sensor = sensor.sort_values(['id', 'date', 'hour'])
sensor.to_csv(path + sensor_file, index = False)
"""