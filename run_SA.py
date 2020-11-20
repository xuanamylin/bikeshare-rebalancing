#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Capstone: Run Simulated Annealing 

Created on Sun Oct 25 16:41:31 2020

@author: xuanlin

"""

import sys
sys.path.append('/Users/xuanlin/Desktop/Capstone/Optimization')
import Divvy_SA as s
from sklearn.externals import joblib
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


# ----------------- VARIABLE ----------------- #

path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
sensor_file = '2018Q12_top25station_sensor.csv'
model_in_name = 'model_in.pkl'
model_out_name = 'model_out.pkl'
traffic_dataset_name = 'modeling_dataset.csv'
travel_distance_name = 'station_distances.csv'
station_info_name = "station_info.csv"


top_station_ids = [35, 192,  91,  77,  43, 133, 174,  81,  76,  90, 177, 287, 268,
                   195,  85, 283, 100,  66, 110,  52, 181,  48,  59, 176,  49]

sample_date = '2018-05-31'
sample_bin = 8 
features = ['month', 'dayofweek', 'time_bin',    # this has to be replaced later
            'in_trips_prev1', 'out_trips_prev1', 
            'in_trips_ystd', 'out_trips_ystd']

time_limit = 120
truck_capacity = 15
debug = False

#time_mtrx = np.array([0, 3, 12, 4,
#                      3, 0, 8, 9,
#                      12, 8, 0, 10,
#                      4, 9, 10, 0]).reshape(4,4)

#actual_list = [0, 7, 3, 4] # current
#expected_list = [0, 3, 7, 3] # preducted
#station_capacity = [0, 8, 8, 8]

hyperparameter = {'temp_schedule': 100, 
                  'K': 100, 
                  'alpha': 0.99, 
                  'iter_max': 10000, 
                  'temp': 100, 
                  'tolerance': 500
                  }



# --------------- Data Processiong Functions ---------------- #

def get_expected_cnt(net_flow, capacity, actual):
    # net inflow
    if net_flow > 0:
        return int(max(capacity - net_flow, 0))
    # net outflow
    elif net_flow < 0:
        return int(min(abs(net_flow), capacity))
    else:
        return int(actual)

def get_expected_cnt_extra(net_flow, capacity):
    # net inflow: (0, capacity - docks needed)
    if net_flow > 0:
        return (0, max(capacity - net_flow, 0))
    # net outflow: (bikes needed, full)
    elif net_flow < 0:
        return (min(abs(net_flow), capacity), capacity)
    # no adjustment needed
    else:
        return (0, capacity)
    
    
# ------------------ PREDICTION ------------------ #

# Load data & model
sensor_df = pd.read_csv(path + sensor_file)
model_inflow = joblib.load(path +  model_in_name)
model_outflow = joblib.load(path +  model_out_name)
traffic = pd.read_csv(path + traffic_dataset_name)
distance = pd.read_csv(path + "station_distances.csv", names = ["from", "to", "distance"])
station_info = pd.read_csv(path + station_info_name)

# Number of docks & station capacity
sensor_info = sensor_df[(sensor_df['date'] == sample_date) & (sensor_df['hour'] == sample_bin)
                        & (pd.to_datetime(sensor_df['date_clean']).dt.minute == 0)][['id', 'available_bikes', 'docks_in_service']]
sensor_info.columns = ['station_id', 'actual_cnt', 'capacity']

# Get distance matrix (min)
time_mtrx = distance[(distance['from'].isin(top_station_ids)) & (distance['to'].isin(top_station_ids))].copy()
time_mtrx = pd.pivot_table(time_mtrx, values='distance', index='from', columns='to', aggfunc=np.sum)
time_mtrx = time_mtrx / 600

# Get expected demand
case = traffic[(traffic['date'] == sample_date) & (traffic['time_bin'] == sample_bin)].copy()
case['inflow_pred'] = model_in.predict(case[features]).round()
case['outflow_pred'] = model_out.predict(case[features]).round()
case['net_flow'] = case['inflow_pred'] - case['outflow_pred']
case = pd.merge(case, sensor_info, on = 'station_id', how = 'outer')
case['expected_cnt'] = case[['net_flow', 'capacity', 'actual_cnt']].apply(lambda x: get_expected_cnt(x[0], x[1], x[2]), axis = 1)
case.sort_values('station_id', inplace = True)


# ------------------ RUN ALGO ------------------ #

# Set up the problem
opt = s.SA(time_mtrx = time_mtrx.values,
         time_limit = time_limit,
         actual_list = list(case['actual_cnt']), 
         expected_list = list(case['expected_cnt']),
         station_capacity = list(case['capacity']),
         truck_capacity = truck_capacity,
         debug = debug)


opt.simulated_annealing()  # Run the optimizatino algorithm
opt.print_solution()  # Print report
opt.plot_convergence(plot_obj_curr=True)



# ------------------ Plot Convergence ------------------ #

from matplotlib import pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

a = moving_average(np.array(opt.progress['obj_curr']), n = 10)

plt.plot(a)



