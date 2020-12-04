#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_opt Small Scale

Created on Thu Nov 19 23:47:42 2020

@author: xuanlin
"""


import sys
sys.path.append('/Users/xuanlin/Desktop/Capstone/Optimization')
import numpy as np
import pandas as pd
import Divvy_SA_multi_truck as sa
import Divvy_ACO_multi as aco
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

features = ['month', 'dayofweek', 'time_bin',    # this has to be replaced later
            'in_trips_prev1', 'out_trips_prev1', 
            'in_trips_ystd', 'out_trips_ystd']

time_limit = 120
truck_capacity = 15
debug = False
start_station = 0
n_truck = 2

sa_hyperparameters = {'n_truck': n_truck,
                      'temp_schedule': 75, 
                      'K': 50, 
                      'alpha': 0.95,
                      'iter_max': 6000, 
                      'temp': 100, 
                      'tolerance': 2000,
                      'do_nothing_punishment': 0,
                      'verbose': False
                      }

aco_hyperparameters = {'num_truck': n_truck,
                       'n_ants': 1,
                       'n_best': 1,
                       'n_iterations': 5000,
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

def case_setup(sensor_data, distance_data, traffic_data, model_inflow, model_outflow, sample_date, sample_bin):
    
    # Number of docks & station capacity
    sensor_info = sensor_data[(sensor_data['date'] == sample_date) & (sensor_data['hour'] == sample_bin)
                            & (pd.to_datetime(sensor_data['date_clean']).dt.minute == 0)][['id', 'available_bikes', 'docks_in_service']]
    sensor_info.columns = ['station_id', 'actual_cnt', 'capacity']
    
    # Get distance matrix (min)
    time_mtrx = distance_data[(distance_data['from'].isin(top_station_ids)) 
                              & (distance_data['to'].isin(top_station_ids))].copy()
    time_mtrx = pd.pivot_table(time_mtrx, values='distance', index='from', columns='to', aggfunc=np.sum)
    time_mtrx = time_mtrx / 600
    
    # Get expected demand
    case = traffic_data[(traffic_data['date'] == sample_date) 
                        & (traffic_data['time_bin'] == sample_bin)].copy()
    case['inflow_pred'] = model_inflow.predict(case[features]).round()
    case['outflow_pred'] = model_outflow.predict(case[features]).round()
    case['net_flow'] = case['inflow_pred'] - case['outflow_pred']
    case = pd.merge(case, sensor_info, on = 'station_id', how = 'outer')
    case['expected_cnt'] = case[['net_flow', 'capacity', 'actual_cnt']].apply(lambda x: get_expected_cnt(x[0], x[1], x[2]), axis = 1)
    case.sort_values('station_id', inplace = True)
    
    return time_mtrx.values, list(case['actual_cnt']), list(case['expected_cnt']), list(case['capacity'])

    

# ------------------ RUN SA ------------------ #

# Helper functions
def convert_ACO_time_mtrx(mtrx):
    N = len(mtrx)
    for i in range(N):
        mtrx[i][i] = np.inf
    return mtrx

# Output result for ACO
def output_aco_solution(aco_output, aco_opt, actual_list, expected_list):
    best_path, time_used, satisfied_customers, satisfied_customers_by_truck, truck_final_inv, demand_left, action, truck_inv = aco_output
    unsatisfied_customers = sum(abs(aco_opt.demand)) - satisfied_customers
    flattened_route = [[p[0] for p in route] + [aco_opt.start_station] for route in best_path]
        
    aco_station_inv_df = pd.DataFrame({'stop': range(len(actual_list)),
                                       'actual': actual_list,
                                       'expected': expected_list,
                                       'redist': np.array(expected_list) + np.array(demand_left),
                                       'diff': abs(demand_left)
                                       })    
    aco_route = {}
    for truck in range(aco_opt.num_truck):
        aco_route_df = pd.DataFrame({'stop': flattened_route[truck],
                                         'truck_inv': truck_inv[truck],
                                         'action': action[truck],
                                         'actual': [actual_list[i] for i in flattened_route[truck]],
                                         'expected': [expected_list[i] for i in flattened_route[truck]]
                                         })
        aco_route_df['redist'] = aco_route_df['actual'] - aco_route_df['action']
        if (action[0] != 0) | (action[-1] != 0):
            if action[0] == 0:
                aco_route_df.iloc[0, -1] = np.nan
            else:
                aco_route_df.iloc[-1, -1] = np.nan
        aco_route[truck] = aco_route_df
    
    return {'time': time_used,
            'satisfied_customers': satisfied_customers,
            'unsatisfied_customers': unsatisfied_customers,
            'route_df': aco_route,
            'station_inv_df': aco_station_inv_df}


def save_pickle(dict_to_save, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        content = pickle.load(handle)
    return content


def run_cases(test_cases, N, filepath_template, data_source_dict, 
              sa_hyperparameters, aco_hyperparameters, version, debug = False):
    
    result_comparison = []
    
    for test in test_cases:
        
        sample_date, sample_bin = test
        time_mtrx, actual_list, expected_list, capacity_list = case_setup(sample_date = sample_date,
                                                                          sample_bin = sample_bin,
                                                                          **data_source_dict)
        
        # SA
        sa_opt = sa.SA(time_mtrx = time_mtrx[:N, :N],
                       time_limit = time_limit,
                       actual_list = actual_list[:N],
                       expected_list = expected_list[:N],
                       station_capacity = capacity_list[:N],
                       truck_capacity = truck_capacity,
                       debug = False,
                       **sa_hyperparameters)
        sa_opt.simulated_annealing()
        sa_solution = sa_opt.output_solution(verbose = False)
        
        # ACO
        aco_opt = aco.Ant_Colony(travel_time = convert_ACO_time_mtrx(time_mtrx)[:N,:N], 
                                 demand = np.array(actual_list)[:N] - np.array(expected_list)[:N],
                                 capacity = truck_capacity,
                                 time_constraint = time_limit,
                                 start_station = start_station, 
                                 **aco_hyperparameters)
        aco_output = aco_opt.run()
        aco_solution = output_aco_solution(aco_output, aco_opt, actual_list[:N], expected_list[:N])
        
        # Aggregate result
        opt_solutions = {'sa': sa_solution, 'aco': aco_solution}
        result_comparison += [{'sample_date': sample_date,
                             'sample_bin': sample_bin,
                             'sa_unsat': sa_solution['unsatisfied_customers'],
                             'sa_time': sa_solution['time'],
                             'aco_unsat': aco_solution['unsatisfied_customers'],
                             'aco_time': aco_solution['time'],
                             }]
    
        pickle_name = filepath_template.format(n_truck, N, sample_date, sample_bin, version)
        save_pickle(opt_solutions, pickle_name)
        
        if debug: print("Case ({}, {}) Done.".format(sample_date, sample_bin))
    
    return pd.DataFrame(result_comparison)
    

# Generate samples
if __name__ == '__main__':
    
    data_sources = {'sensor_data': sensor_df,
                   'distance_data': distance,
                   'traffic_data': traffic,
                   'model_inflow': model_inflow,
                   'model_outflow': model_outflow
                   }
    
    test_cases = [('2018-05-31', 8), ('2018-04-01', 10), ('2018-04-18', 16), ('2018-03-30', 18), 
                  ('2018-02-18', 12), ('2018-05-02', 12), ('2018-03-21', 14), ('2018-02-01', 8),
                  ('2018-01-29', 18), ('2018-04-20', 14)]
    
    #inpt2 = [('2018-05-30',6), ('2018-05-30',8), ('2018-05-30',10),('2018-05-30',12)]
    #inpt3 = [('2018-05-29',6), ('2018-05-29',8), ('2018-05-29',10),('2018-05-29',12)]
    
    filepath_template = output_path + "solution_{}truck_{}stops_{}_{}_{}.pickle"
    
    # Run and Save Results
    N = 18
    version = 1
    result_df = run_cases(test_cases, N, filepath_template, data_sources, 
                          sa_hyperparameters, aco_hyperparameters, version, debug = True)
    result_df.to_csv(output_path + '{}truck_{}stop_agg_{}.csv'.format(n_truck, N, version), index = False)

