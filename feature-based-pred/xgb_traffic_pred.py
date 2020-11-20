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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)


# Input --------------------------------------

path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
trip_file = '2018Q12_top25station_trips.csv'
sensor_file = '2018Q12_top25station_sensor.csv'
inflow_pred_to_save = '2018Q12_top25station_inflow_pred.csv'
outflow_pred_to_save = '2018Q12_top25station_outflow_pred.csv'

window_duration = 2 # hour
operating_window = [6, 20] # remove data from time_a to time_b
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
print("Sensor file size: {}".format(sensor.shape))  # sensor.tail()


# Y labels ---------------------------------------------

keys = ['station_id', 'date', 'time_bin']

# y labels: outflows
out_df = trips[trips['from_station_id'].isin(top_station_ids)].copy()
out_df['start_time'] = pd.to_datetime(out_df['start_time'])
out_df['start_date'] = out_df['start_time'].dt.date
out_df['start_time_bin'] = pd.cut(out_df['start_time'].dt.hour,
                                 bins = range(-1, 25, window_duration),
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
                                 bins = range(-1, 25, window_duration),
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

# weather & temperature

# inflows & outflows of nearby stops (radius)

# existing number of bikes & docks in station

# whether it's a national holiday / chicago holiday

# Fillna
dataset['time_bin'] = list(map(int, dataset['time_bin']))  # convert category to int for convenience
dataset.fillna(-999, inplace = True)



# Filtering -----------------------------------------------------------------

# 1. Limit data to opearting hours
#dataset = dataset[(dataset['time_bin'] >= operating_window[0])
#                  & (dataset['time_bin'] <= operating_window[1]-2)]  # (31675, 11)

# 2. Eliminate data during outage

# Outage periods collected by sensors
bike_outage = sensor[sensor['available_bikes'] == 0][['id', 'date_clean']].copy()
bike_outage.columns = ['station_id', 'date_clean']

dock_outage = sensor[sensor['available_docks'] == 0][['id', 'date_clean']].copy()
dock_outage.columns = ['station_id', 'date_clean']

print('Bike outage periods: {}'.format(bike_outage.shape[0]))  # 16726
print('Dock outage periods: {}'.format(dock_outage.shape[0]))  # 8907

# Convert to time_bins
bike_outage['date'] = pd.to_datetime(bike_outage['date_clean']).dt.date
bike_outage['hour'] = pd.to_datetime(bike_outage['date_clean']).dt.hour
bike_outage['time_bin'] = pd.cut(bike_outage['hour'],
                                 bins = range(-1, 25, window_duration),
                                 include_lowest = False,
                                 labels = range(0, 23, window_duration))
bike_outage['time_bin'] = list(map(int, bike_outage['time_bin']))
                                 

dock_outage['date'] = pd.to_datetime(dock_outage['date_clean']).dt.date
dock_outage['time_bin'] = pd.cut(pd.to_datetime(dock_outage['date_clean']).dt.hour,
                                 bins = range(-1, 25, window_duration),
                                 include_lowest = True,
                                 labels = range(0, 23, window_duration))
dock_outage['time_bin'] = list(map(int, dock_outage['time_bin']))

# Filter out bottleneck time_bins
in_dataset = pd.merge(dataset, dock_outage[['station_id', 'date', 'time_bin']].drop_duplicates(),
                 on = ['station_id', 'date', 'time_bin'], how = 'outer', indicator = True)
in_dataset = in_dataset[in_dataset['_merge'] == 'left_only']

out_dataset = pd.merge(dataset, bike_outage[['station_id', 'date', 'time_bin']].drop_duplicates(),
                 on = ['station_id', 'date', 'time_bin'], how = 'outer', indicator = True)
out_dataset = out_dataset[out_dataset['_merge'] == 'left_only']

in_dataset.set_index(['station_id', 'date'], inplace = True)
out_dataset.set_index(['station_id', 'date'], inplace = True)


print("Inflow dataset: {}".format(in_dataset.shape))  # (29324, 12)
print("Outflow dataset: {}".format(out_dataset.shape))  # (28044, 12)



# Modeling --------------------------------------------

def train_n_pred(dataset, features, label, stratify_col, test_size=0.2, random_state=2020):
    # Predict inflow 
    X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[label],
                                                        stratify = dataset[stratify_col],
                                                        test_size = test_size, random_state = random_state)
    print('Train set size: {}, {}'.format(X_train.shape, y_train.shape))
    print('Test set size: {}, {}'.format(X_test.shape, y_test.shape))
    
    # Train & Predict
    model = xgb.XGBRegressor()
    print("\nTrain Model...")
    model.fit(X_train, y_train)
    print("Predict...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluation
    metrics = pd.DataFrame({'r2': [r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)],
                           'rmse': [np.sqrt(mean_squared_error(y_train, y_pred_train)), 
                                    np.sqrt(mean_squared_error(y_test, y_pred_test))],
                           'mae': [median_absolute_error(y_train, y_pred_train), 
                                   median_absolute_error(y_test, y_pred_test)]
                           }, index = ['train', 'test'])
    print("\nEvaluation for Train & Test")
    print(metrics)
    
    X_train['y_true'], X_train['y_pred'] = y_train, y_pred_train
    X_test['y_true'], X_test['y_pred'] = y_test, y_pred_test
    
    return X_train, X_test, model, metrics



# Parameters
features = ['month', 'dayofweek', 'time_bin', 
            'in_trips_prev1', 'out_trips_prev1', 
            'in_trips_ystd', 'out_trips_ystd']

stratify_col = ['month', 'dayofweek', 'time_bin']

params = {'features': features, 'stratify_col': stratify_col, 'random_state': random_state}

# Inflow model
train_in, test_in, model_in, metrics_in = train_n_pred(dataset = in_dataset, label = 'in_trips', **params)
#path + inflow_pred_to_save

# Outflow model 
train_out, test_out, model_out, metrics_out = train_n_pred(dataset = out_dataset, label = 'out_trips', **params)


# Predict for a Sample
sample_date = '2018-05-31'
sample_bin = 8
sample_testset = dataset[(dataset['date'] == pd.to_datetime('2018-05-31')) &
                         (dataset['time_bin'] == 8)]
sample_testset['inflow_pred'] = model_in.predict(sample_testset[features]).round()
sample_testset['outflow_pred'] = model_out.predict(sample_testset[features]).round()

dataset.to_csv(path + 'modeling_dataset.csv', index = False)

from sklearn.externals import joblib
joblib.dump(model_in, path + 'model_in.pkl')
joblib.dump(model_out, path + 'model_out.pkl')


"""
## limit to operating hours ##
in
             r2      rmse       mae
train  0.780069  7.529283  2.483760
test   0.700070  8.559241  2.534907

out
             r2      rmse       mae
train  0.795729  6.548119  2.193581
test   0.765157  6.810390  2.191537


## Without limiting operating hours ##
Evaluation for Train & Test
in
             r2      rmse       mae
train  0.779726  6.085722  1.206968
test   0.755948  6.098199  1.199870

out
             r2      rmse       mae
train  0.800689  5.269502  1.153267
test   0.752593  5.826440  1.181848

"""


# Prediction Analysis -------------------------------------

from matplotlib import pyplot as plt

# Plot distribution of y labels and predicted y labels
# inflow
in_dist = dataset['in_trips'].value_counts(normalize=True).reset_index()
in_dist.columns = ['n_trips', 'share']
plt.figure()
plt.bar(in_dist['n_trips'].head(30), in_dist['share'].head(30))
plt.title('Distribution of Inflows (window = 2hrs)')
plt.xlabel("Inflow")
plt.show()

# outflow
out_dist = dataset['out_trips'].value_counts(normalize=True).reset_index()
out_dist.columns = ['n_trips', 'share']
plt.figure()
plt.bar(out_dist['n_trips'].head(30), out_dist['share'].head(30))
plt.title('Distribution of Outflows (window = 2hrs)')
plt.xlabel("Outflow")
plt.show()


# Plot difference of distribution between y_true and y_pred
def plot_pred_dist(y_true, y_pred): # inputs are pd.Series
    true_dist = y_true.value_counts(normalize=True).reset_index()
    true_dist.columns = ['n_trips', 'share']
    pred_dist = y_pred.apply(round).value_counts(normalize=True).reset_index()
    pred_dist.columns = ['n_trips', 'share']
    
    plt.figure(figsize = (8, 4))
    plt.bar(np.array(true_dist['n_trips'].head(25))-0.2, true_dist['share'].head(25), width = 0.4)
    plt.bar(np.array(pred_dist['n_trips'].head(25))+0.2, pred_dist['share'].head(25), width = 0.4)
    plt.title('Distribution of Inflows (window = 2hrs)')
    plt.xlabel("Inflow")
    plt.legend(["y_test", "y_pred"])
    plt.show()
    
plot_pred_dist(test_in['y_true'], test_in['y_pred'])


def group_metrics(row):
    return pd.Series({'r2': r2_score(row['y_true'], row['y_pred']), 
                      'rmse': np.sqrt(mean_squared_error(row['y_true'], row['y_pred'])),
                      'mae': median_absolute_error(row['y_true'], row['y_pred'])
                      })

# Evaluation: Day of Week
dow_metrics = test_in.groupby('dayofweek').apply(group_metrics).reset_index()
plt.figure()
plt.plot(dow_metrics.iloc[:,1:])
plt.legend(dow_metrics.columns[1:])
plt.title("Metrics By Day of Weey")
plt.xticks(range(7), range(1,8))
plt.xlabel("Day of Week")
plt.show()


# Evaluation: Time Bin
bin_metrics = test_in.groupby('time_bin').apply(group_metrics).reset_index()
N = bin_metrics.shape[0]
plt.figure()
plt.plot(bin_metrics.iloc[:,1:])
plt.legend(bin_metrics.columns[1:])
plt.title("Metrics By Time Window")
plt.xticks(range(N), bin_metrics.iloc[:,0])
plt.xlabel("Time Window (hr)")
plt.show()


# Evaluation: Month
m_metrics = test_in.groupby('month').apply(group_metrics).reset_index()
N = m_metrics.shape[0]
plt.figure()
plt.plot(m_metrics.iloc[:,1:])
plt.legend(m_metrics.columns[1:])
plt.title("Metrics By Mon")
plt.xticks(range(N), m_metrics.iloc[:,0])
plt.xlabel("Month")
plt.show()

