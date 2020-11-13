#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Capstone - XGBoost Time Series Model

Created on Sun Nov  8 01:25:50 2020

@author: xuanlin
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)


# Input --------------------------------------

path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
trip_file = '2018Q12_top25station_trips.csv'
sensor_file = '2018Q12_top25station_sensor.csv'

window_duration = 2 # hour
cutout_hour = [2, 5] # remove data from time_a to time_b
top_station_ids = [35, 192,  91,  77,  43, 133, 174,  81,  76,  90, 177, 287, 268,
                   195,  85, 283, 100,  66, 110,  52, 181,  48,  59, 176,  49]

random_state = 2020
# Question: does trips and station_historicals match up?


# Load Data --------------------------------------------

trips = pd.read_csv(path + trip_file)
print("Trip file size: {}".format(trips.shape))
#['trip_id', 'start_time', 'end_time', 'bike_id', 'trip_duration',
#'from_station_id', 'to_station_id', 'usertype', 'gender', 'birth_year',
#'start_time_bin', 'start_date'],
      
sensor = pd.read_csv(path + sensor_file)
print("Sensor file size: {}".format(sensor.shape))


# Y labels

keys = ['station_id', 'date', 'time_bin']

# y labels: outflows
out_df = trips[trips['from_station_id'].isin(top_station_ids)].copy()
out_df['start_time'] = pd.to_datetime(out_df['start_time'])
out_df['start_date'] = out_df['start_time'].dt.date
out_df['start_time_bin'] = pd.cut(out_df['start_time'].dt.hour,
                                 bins = range(0, 25, window_duration),
                                 include_lowest = True,
                                 labels = range(0, 23, window_duration))  #.astype('category').cat.codes * 2
out_df = out_df.groupby(['from_station_id', 'start_date', 'start_time_bin'])['trip_id'].count().reset_index()
out_df.columns =keys + ['out_trips']
print('Outflow DF shape: {}'.format(out_df.shape)) # Outflow DF shape: (34492, 4) / Outflow DF shape: (54300, 4)


# y labels: inflows
in_df = trips[trips['to_station_id'].isin(top_station_ids)].copy()
in_df['end_time'] = pd.to_datetime(in_df['end_time'])
in_df['end_date'] = in_df['end_time'].dt.date
in_df['end_time_bin'] = pd.cut(in_df['end_time'].dt.hour,
                                 bins = range(0, 25, window_duration),
                                 include_lowest = True,
                                 labels = range(0, 23, window_duration))  #.astype('category').cat.codes * 2
in_df = in_df.groupby(['to_station_id', 'end_date', 'end_time_bin'])['trip_id'].count().reset_index()
in_df.columns = keys + ['in_trips']
print('Inflow DF shape: {}'.format(in_df.shape))  # Inflow DF shape: (34433, 4) / Inflow DF shape: (54300, 4)

# Join in flows and outflows
dataset = pd.merge(in_df, out_df, how = 'outer', on = keys)
dataset = dataset.sort_values(['station_id', 'date', 'time_bin'])
dataset.reset_index(drop = True, inplace = True)
print('Combined DF shape: {}'.format(dataset.shape))  # Combined DF shape: (54300, 5)
print('Unique station ID count: {}'.format(len(dataset['station_id'].unique())))


# Feature Engineering --------------------------------

# day of week
dataset['dayofweek'] = pd.to_datetime(dataset['date']).dt.dayofweek + 1

# month
dataset['month'] =list(map(int, pd.to_datetime(dataset['date']).dt.month))


# in and out in the previous window
dataset[['in_trips_prev1', 'out_trips_prev1']] = dataset[['in_trips', 'out_trips']].shift(periods = 1)
dataset.loc[[False] + list(dataset['station_id'][1:].values != dataset['station_id'][:-1].values),
        ['in_trips_prev1', 'out_trips_prev1']] = np.nan


# in and out yesterday in the same window
ystd_period = int(24 / window_duration)
dataset[['in_trips_ystd', 'out_trips_ystd']] = dataset[['in_trips', 'out_trips']].shift(periods = ystd_period)
dataset.loc[[False] + list(dataset['station_id'][1:].values != dataset['station_id'][:-1].values),
        ['in_trips_ystd', 'out_trips_ystd']] = np.nan


# yesterday's total inflows & outflows

# weather

# inflows & outflows of nearby stops


# Fillna
dataset['time_bin'] = list(map(int, dataset['time_bin']))  # convert category to int for convenience
dataset.fillna(-999, inplace = True)

print("Final dataset size: {}".format(dataset.shape))


# Modeling --------------------------------------------



features = ['month', 'dayofweek', 'time_bin', 
            'in_trips_prev1', 'out_trips_prev1', 
            'in_trips_ystd', 'out_trips_ystd']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset['in_trips'],
                                                    stratify = dataset[['month', 'dayofweek', 'time_bin']],
                                                    test_size = 0.2, random_state = random_state)
print('Train set size: {}, {}'.format(X_train.shape, y_train.shape))
print('Test set size: {}, {}'.format(X_test.shape, y_test.shape))

# Train & Predict
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluation
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = median_absolute_error(y_test, y_pred)
print('R2: {}\nMSE: {}\nMAE: {}'.format(r2, mse, mae))






