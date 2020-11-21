#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Capstone: Run Simulated Annealing 

Created on Sun Oct 25 16:41:31 2020

@author: xuanlin

"""

import sys
sys.path.append('/Users/xuanlin/Desktop/Capstone/Optimization')
import numpy as np
import pandas as pd
import Divvy_SA as sa
import Divvy_ACO as aco
from sklearn.externals import joblib
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
import pickle


# ----------------- VARIABLE ----------------- #

path = '/Users/xuanlin/Desktop/Capstone/Optimization/data/'
output_path = '/Users/xuanlin/Desktop/Capstone/Optimization/output/'
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
start_station = 0

sa_hyperparameters = {'temp_schedule': 80, 
                  'K': 50, 
                  'alpha': 0.8,
                  'iter_max': 5000, 
                  'temp': 100, 
                  'tolerance': 1000
                  }

aco_hyperparameters = {'n_ants': 1,
                      'n_best': 1,
                      'n_iterations': 500,
                      'decay': 0.5,
                      'alpha': 0.2,
                      'beta': 1}

# ---------------- Data Processiong Functions ---------------- #

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
case['inflow_pred'] = model_inflow.predict(case[features]).round()
case['outflow_pred'] = model_outflow.predict(case[features]).round()
case['net_flow'] = case['inflow_pred'] - case['outflow_pred']
case = pd.merge(case, sensor_info, on = 'station_id', how = 'outer')
case['expected_cnt'] = case[['net_flow', 'capacity', 'actual_cnt']].apply(lambda x: get_expected_cnt(x[0], x[1], x[2]), axis = 1)
case.sort_values('station_id', inplace = True)


# ------------------ RUN SA ------------------ #

# Set up the problem
sa_opt = sa.SA(time_mtrx = time_mtrx.values,
             time_limit = time_limit,
             actual_list = list(case['actual_cnt']), 
             expected_list = list(case['expected_cnt']),
             station_capacity = list(case['capacity']),
             truck_capacity = truck_capacity,
             do_nothing_punishment = 0,
             debug = debug,
             **sa_hyperparameters)

sa_opt.simulated_annealing()  # Run the optimizatino algorithm
sa_solution = sa_opt.output_solution(verbose = False)  # Print report

sa_opt.plot_convergence(plot_obj_curr=False)

sa_solution['time']
sa_solution['unsatisfied_customers'] # 279, 262, 260, 259, 234


# ------------------ RUN ACO ----------------- #

# Helper functions
def convert_ACO_time_mtrx(mtrx):
    N = len(mtrx)
    for i in range(N):
        mtrx[i][i] = np.inf
    return mtrx

def output_aco_solution(aco_output, aco_opt, actual_list, expected_list):
    best_path, time_used, satisfied_customers, truck_final_inv, redist_cnt, action, truck_inv = aco_output
    unsatisfied_customers = sum(abs(aco_opt.demand)) - satisfied_customers
    aco_route = [p[0] for p in best_path] + [start_station]
        
    aco_station_inv_df = pd.DataFrame({'stop': range(len(actual_list)),
                                       'actual': actual_list,
                                       'expected': expected_list,
                                       'redist': np.array(actual_list) - np.array(redist_cnt)
                                       })
    
    aco_route_df = pd.DataFrame({'stop': aco_route,
                                 'truck_inv': truck_inv,
                                 'action': action,
                                 'actual': [aco_station_inv_df['actual'][i] for i in aco_route],
                                 'expected': [aco_station_inv_df['expected'][i] for i in aco_route]
                                 })
    
    #route_redist = aco_route_df['actual'] - aco_route_df['action']
    aco_route_df['redist'] = aco_route_df['actual'] - aco_route_df['action']
    if (action[0] != 0) | (action[-1] != 0):
        if action[0] == 0:
            aco_route_df.iloc[0, -1] = np.nan
        else:
            aco_route_df.iloc[-1, -1] = np.nan
    #aco_route_df['redist'] = route_redist
    
    return {'time': time_used,
            'satisfied_customers': satisfied_customers,
            'unsatisfied_customers': unsatisfied_customers,
            'route_df': aco_route_df,
            'station_inv_df': aco_station_inv_df}


# Run algorithm
actual_list = list(case['actual_cnt'])
expected_list = list(case['expected_cnt'])
aco_opt = aco.Ant_Colony(travel_time = convert_ACO_time_mtrx(time_mtrx.values), 
                 demand = np.array(case['actual_cnt'] - case['expected_cnt']),
                 capacity = truck_capacity, 
                 time_constraint = time_limit,
                 start_station = start_station, 
                 **aco_hyperparameters)

aco_output = aco_opt.run()
aco_solution = output_aco_solution(aco_output, aco_opt, actual_list, expected_list)



# ----------------- SAVE RESULTS ----------------- #

def save_pickle(dict_to_save, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        content = pickle.load(handle)
    return content

opt_solutions = {'sa': sa_solution, 'aco': aco_solution}

pickle_name = "solution_{}_{}_{}.pickle".format(sample_date, sample_bin, 1)
save_pickle(opt_solutions,output_path + pickle_name)




"""
# ------------------ Plot Convergence ------------------ #

from matplotlib import pyplot as plt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

a = moving_average(np.array(opt.progress['obj_curr']), n = 10)

plt.plot(a)
"""


