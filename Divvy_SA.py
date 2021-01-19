#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Simulated Annealing for Divvy

Created on Fri Oct 16 22:15:26 2020

@author: xuanlin

"""

import numpy as np
import pandas as pd
import random as r
import datetime as dt
from matplotlib import pyplot as plt
from operator import itemgetter
from itertools import chain, groupby
from numpy.random import choice, shuffle
import warnings


# Timing Decorator
def time_it(func):
    def wrapped_func(self):
        start_time = dt.datetime.now()
        print("Start time: {}".format(start_time))
        func(self)
        end_time = dt.datetime.now()
        print("End time: {}".format(end_time))
        print("Time taken: {}\n".format(end_time - start_time))
    return wrapped_func


class ProblemDefinitionError(Exception):
    pass


class SA():
    def __init__(self, 
                 n_truck,
                 time_mtrx, 
                 actual_list, 
                 expected_list, 
                 station_capacity, 
                 truck_capacity, 
                 time_limit = 1000,
                 temp_schedule=100, 
                 K=100, 
                 alpha=0.99, 
                 iter_max=1000, 
                 temp=100, 
                 tolerance=500, 
                 punish_do_nothing=True, 
                 do_nothing_punishment = 1000, 
                 multi_visit = True,
                 track_progress=True, verbose=True, debug=False):
        
        # depot is assumed to be the first stop  
        # SA algorithm hyper-parameters
        self.n_truck = n_truck
        self.K = K
        self.temp_schedule = temp_schedule
        self.alpha = alpha
        self.iter_max = iter_max
        self.temp = temp
        self.tolerance = tolerance
        self.verbose = verbose
        self.punish_do_nothing = punish_do_nothing
        self.do_nothing_punishment = do_nothing_punishment
        self.track_progress = track_progress
        if self.track_progress: self.progress = []
        self.multi_visit = multi_visit
        self.debug = debug
        
        # Problem constants
        self.ind_to_stop, self.actual_list, self.expected_list, self.time_mtrx, self.station_capacity \
            = self.preprocess_constants(actual_list, expected_list, time_mtrx, station_capacity)
        self.actual_list_raw = actual_list
        self.expected_list_raw = expected_list
        self.station_capacity_raw = station_capacity
        self.C = truck_capacity
        self.time_limit = time_limit
        
        self.N = len(self.actual_list)
        output_action = lambda act, exp: 1 if act < exp else -1 if act > exp else 0
        
        # p_action: pickup(-1) or dropoff(1)
        self.p_action = [output_action(x[0], x[1]) for x in zip(self.actual_list, self.expected_list)]
        
        
        self.diff = list(np.array(self.expected_list) - np.array(self.actual_list))
        
        # Sanity check on problem definition
        if any([x[0] > x[1] for x in zip(self.actual_list, self.station_capacity)]):
            raise ProblemDefinitionError("Actual # bikes at stations exceeds capacity.")
        elif any([x[0] > x[1] for x in zip(self.expected_list, self.station_capacity)]):
            raise ProblemDefinitionError("Expected # bikes at stations exceeds capacity.")
        elif all([x == 0 for x in self.diff]):
            raise ProblemDefinitionError("No re-distribution is needed.")
        elif all([x >= 0 for x in self.diff]):
            raise ProblemDefinitionError("Net bike deficit.")
        elif max(chain(*self.time_mtrx)) * 2 > self.time_limit:
            warnings.warm("Warning: Some stops will never be traversed due to the time limit. Consider loosening the time limit.")
    
    
    # Remove stops w/o the need of rebalancing
    def preprocess_constants(self, actual_list, expected_list, time_mtrx, station_capacity):
        ind_to_opt = [0] + list(np.where(np.array(actual_list)[1:] != np.array(expected_list)[1:])[0]+1)
        ind_to_stop = list(itemgetter(*ind_to_opt)(list(range(len(actual_list)))))
        actual_adj = list(itemgetter(*ind_to_opt)(actual_list))
        expected_adj = list(itemgetter(*ind_to_opt)(expected_list))
        time_mtrx_adj = time_mtrx[ind_to_opt,:][:,ind_to_opt]
        station_cap_adj = list(itemgetter(*ind_to_opt)(station_capacity))
        return ind_to_stop, actual_adj, expected_adj, time_mtrx_adj, station_cap_adj
    
    
    # Calculate total time of the route
    def cost(self, seq):
        cost_list = []
        for s in seq:
            segments = [s[i:i+2] for i in range(len(s)-1)]
            time_sum = sum([self.time_mtrx[seg[0], seg[1]] for seg in segments])
            cost_list += [time_sum]
        return cost_list
    
    
    # Revised
    # Re-arrange stops by arrival time
    def seq_by_arrival(self, seq):
        
        arrivals = [] # list of tuples: (truck_id, stop_id, arrival_time)
        
        for truck_id, s in enumerate(seq):
            segments = [s[i:i+2] for i in range(len(s)-1)]
            eta = [0] + list(np.cumsum([self.time_mtrx[seg[0], seg[1]] for seg in segments]))
            arrivals += list(zip([truck_id] * len(s), s, eta))
        
        arrivals = sorted(arrivals, key = lambda x: x[2])  # sort by arrival time
        
        return arrivals

    # revised: generate actions, truck_inv, and station_inv for one iteration
    def gen_actions(self, seq):

        station_inv = self.actual_list.copy()
        actions = [[] for n in range(self.n_truck)]
        truck_inv = [[] for n in range(self.n_truck)]
        seq_ordered = self.seq_by_arrival(seq)
        diff_curr = self.diff.copy()
        
        # The first station, station_id = 0
        if diff_curr[0] < 0:
            #loc_diff = self.diff[0]
            for i in range(self.n_truck):
                to_pickup = max(diff_curr[0], -self.C)       # if pick up, assign to truck 0. the rest at station 0, assign to truck 1
                actions[i] = [to_pickup]
                truck_inv[i] = [-to_pickup]
                station_inv[0] += to_pickup
                diff_curr[0] -= to_pickup
        else:
            for i in range(self.n_truck):
                actions[i] = [0]
                truck_inv[i] = [0]
        
        #seq_ordered_non_0 = [s for s in seq_ordered if s[1] != 0]  # remove the start and the end
        start_tuples = [(n, 0, 0) for n in range(self.n_truck)]
        seq_ordered_non_0 = [s for s in seq_ordered if s not in start_tuples]
        
        for truck, stop, _ in seq_ordered_non_0:
            
            # calculate action
            if diff_curr[stop] < 0:    # pick up, to_pickup < 0
                to_change = max(diff_curr[stop], truck_inv[truck][-1]-self.C) 
                
            elif diff_curr[stop] > 0:  # drop off
                to_change = min(diff_curr[stop], truck_inv[truck][-1])
            else:                      # no action needed
                to_change = 0
                
            actions[truck] += [to_change]
            truck_inv[truck] += [truck_inv[truck][-1] - to_change]
            station_inv[stop] += to_change
            diff_curr[stop] -= to_change
        
        """
        # the last stop, station_id = 0
        if diff_curr[0] <= 0:   # pick up
            for i in range(self.n_truck):
                actions[i] += [0]    # already taken care of in the beginning. Do nothing.
                truck_inv[i] += [truck_inv[i][-1]]
        else:
            #loc_diff = self.diff[0]
            for i in range(self.n_truck):
                to_dropoff = min(diff_curr[0], truck_inv[i][-1])
                actions[i] += [to_dropoff]
                truck_inv[i] += [truck_inv[i][-1] - to_dropoff]
                station_inv[0] += to_dropoff
                diff_curr[0] -= to_dropoff
        """
        
        return actions, truck_inv, station_inv
    
    
    # revised
    def objective(self, seq):
        
        actions, _, station_inv = self.gen_actions(seq)
        
        obj = abs(np.array(station_inv) - np.array(self.expected_list)).sum()
        
        actions_during_trip = chain(*[a[1:-1] for a in actions])  # chop off the start and the end
        if self.punish_do_nothing:
            do_nothing = any([x == 0 for x in actions_during_trip])
            obj += do_nothing * self.do_nothing_punishment
        
        return obj
        
    # revised
    def gen_init_seq(self):
        # get a list of stations that needs to be picked up
        # if there's pickup to be done, assign to Truck 1
        need_pickup = set(np.where(np.array(self.p_action[1:]) == -1)[0] + 1)
        
        if self.p_action[0] == 0: # the first stop needs drop off
            selected_stops = choice(list(need_pickup), self.n_truck, replace = False)
            routes = [[0] + [stop] + [0] for stop in selected_stops]
        else:
            selected_stops = choice(list(need_pickup), self.n_truck-1, replace = False)
            routes = [[0] + [stop] + [0] for stop in selected_stops]
            eligible_stops_left = set(range(1, self.N)).difference(selected_stops)
            select_stop_1st_truck = choice(list(eligible_stops_left), 1)[0]
            routes = [[0] + [select_stop_1st_truck] + [0]] + routes
            
        #print(routes)
        return routes


    def permute_seq(self, seq, perm, stops_to_add = None, truck_to_remove = None, debug=False):
        
        seq_new = seq[:]
        
        if perm == "insert":
            
            if self.multi_visit:
                stop, truck, pos, seq_new = self.insert_stop_multivisit(seq_new, stops_to_add)
            else:
                stops_visited = set(chain(*seq_new))
                stops_to_add = list(set(range(self.N)).difference(stops_visited))
                stop = choice(stops_to_add)
                truck = choice(range(self.n_truck))
                pos_to_add = range(1,len(seq_new[truck]))
                pos = choice(pos_to_add) # add before it
                seq_new[truck] = seq_new[truck][:pos] + [stop] + seq_new[truck][pos:]
                
            if debug: print("Insert: truck - {}, pos - {}, stop - {}".format(truck, pos, stop))
            
            
        elif perm == "remove":
            if truck_to_remove:
                truck = truck_to_remove
            else:
                remove_eligible = [i for i in range(self.n_truck) if len(seq[i]) > 3]
                truck = choice(remove_eligible)
            pos_to_remove = range(1, len(seq_new[truck])-1)
            pos = choice(pos_to_remove)
            seq_new[truck] = seq_new[truck][:pos] + seq_new[truck][pos+1:]
            seq_new[truck] = [x[0] for x in groupby(seq_new[truck])]
            if debug: print("Remove: truck - {}, pos - {}".format(truck, pos))
            
        elif perm == "swap":
            #same_route_swap_ineligible = [i for i in range(self.n_truck) if len(seq[i]) < 4]
            same_route_swap_ineligible = np.where(np.array([len(s) for s in seq]) < 4)[0]
            # Select trucks
            if len(same_route_swap_ineligible) == 0:  # if all trucks are eligible for same-route swap
                truck1, truck2 = choice(range(self.n_truck), 2, replace = True)  # sample w/ replacement from all trucks
            else:
                truck1 = choice(range(self.n_truck))
                truck2_eligible = set(range(self.n_truck)).difference(set(same_route_swap_ineligible))
                truck2 = choice(list(truck2_eligible))
            # Select stops
            if self.multi_visit:
                pos1 = choice(range(1, len(seq_new[truck1])-1))
                pos2_eligible = np.where(np.array(seq_new[truck2][1:-1]) != seq_new[truck1][pos1])[0]+1
                pos2 = choice(pos2_eligible)
                
            else:
                if truck1 == truck2:
                    pos_to_swap = range(1, len(seq_new[truck1])-1)
                    pos1, pos2 = sorted(choice(pos_to_swap, 2, replace = False))
                    #seq_new[truck1] = seq_new[truck1][:pos1] + [seq_new[truck1][pos2]] + seq_new[truck1][pos1+1:pos2] \
                    #                  + [seq_new[truck1][pos1]] + seq_new[truck1][pos2+1:]
                else:
                    pos1 = choice(range(1, len(seq_new[truck1])-1))
                    pos2 = choice(range(1, len(seq_new[truck2])-1))
               
            stop1, stop2 = seq_new[truck1][pos1], seq_new[truck2][pos2]
            seq_new[truck1] = seq_new[truck1][:pos1] + [stop2] + seq_new[truck1][pos1+1:]
            seq_new[truck2] = seq_new[truck2][:pos2] + [stop1] + seq_new[truck2][pos2+1:]
            if self.multi_visit: 
                seq_new[truck1] = [x[0] for x in groupby(seq_new[truck1])]
                seq_new[truck2] = [x[0] for x in groupby(seq_new[truck2])]
            if debug: print("Swap: (truck, pos) 1 - {}, 2 - {}".format((truck1, pos1), (truck2, pos2)))
            
        elif perm == "revert":
            same_route_swap_ineligible = [i for i in range(self.n_truck) if len(seq[i]) < 4]
            truck_eligible = set(range(self.n_truck)).difference(set(same_route_swap_ineligible))
            truck = choice(list(truck_eligible))
            pos_to_revert = range(1, len(seq_new[truck])-1)
            pos1, pos2 = sorted(choice(pos_to_revert, 2, replace = False))
            seq_new[truck] = seq[truck][:pos1] + seq[truck][pos1:pos2+1][::-1] + seq[truck][pos2+1:]
            if self.multi_visit: seq_new[truck] = [x[0] for x in groupby(seq_new[truck])]
            if debug: print("Revert: truck - {}, pos - {}".format(truck, [pos1, pos2]))
       
        else:
            raise ValueError("Illegal permutation. 4 legal permutations include 'insert', 'remove', 'swap', and 'revert'.")
            
        return seq_new
    
    
    def insert_stop_multivisit(self, seq, stops_to_add):
        seq_new = seq[:]
        shuffle(stops_to_add)
        for stop in stops_to_add:
            pos_to_add = [np.where((np.array(s)!=stop) & (np.array([0] + s[:-1])!=stop))[0][1:] for s in seq]
            truck_to_choose = np.where(np.array([len(p) for p in pos_to_add]) > 0)[0]
            if len(truck_to_choose) > 0:
                truck = choice(truck_to_choose)
                pos = choice(pos_to_add[truck])
                seq_new[truck] = seq_new[truck][:pos] + [stop] + seq_new[truck][pos:]
                break
        return stop, truck, pos, seq_new
    
    
    
    
    def gen_new_seq(self, seq, debug=False):
        
        stops_visited = list(chain(*seq))
        cost_list = self.cost(seq)
        truck_to_remove = None
        stops_to_add = None
        
        if self.multi_visit:
            actions, truck_inv, station_inv = self.gen_actions(seq)
            diff = np.array(self.expected_list) - np.array(station_inv)
            ineligible_stops = {0}
            # avoid situation: add 2 to [[0,2,0], [0,2,0]]
            if len(list(chain(*seq))) == 3 * self.n_truck:
                stop_between = set(np.array(seq)[:,1])
                if len(stop_between) == 1:
                    ineligible_stops = ineligible_stops.union(stop_between)
            stops_to_add = list(set(np.where(np.array(diff) != 0)[0]).difference(ineligible_stops))
            demand_left_bool = (len(stops_to_add) > 0) & (not all(diff > 0)) & (not all(diff < 0))
        else:
            demand_left_bool = None # not used if self.multi_visit = False
        
        # Decide on candidate permutations to perform
        # exceed max time -> do not add
        if max(cost_list) > self.time_limit:
            # Arguably, a solution exceeding the time_limit wouldn't be output at the first place
            candidate_perm = ['remove', 'swap', 'revert']
            truck_to_remove = np.where(np.array(cost_list) > self.time_limit)[0]
            if len(truck_to_remove) > 0:
                truck_to_remove = truck_to_remove[0]  # remove one at a time
                
        # single visit + all stops included -> then don't add or remove
        elif (~self.multi_visit) & (len(set(stops_visited)) == self.N):
            candidate_perm = ['swap', 'revert']
        # multi visit + no demand left --> don't add or remove
        elif self.multi_visit & (not demand_left_bool):
                candidate_perm = ['swap', 'revert']
            
        # each route only has one stop --> do not remove, swap, or remove
        elif len(stops_visited) == self.n_truck * 3:
            candidate_perm = ['insert']
            
        # otherwise, all permutations and legal
        else:
            candidate_perm = ['insert', 'remove', 'swap', 'revert']
        
        if debug:
            if self.multi_visit:
                print("diff: {}, demand_left_bool: {}, candidate_perm: {}, time: {}".format(diff, demand_left_bool, candidate_perm, max(cost_list)))
            else:
                print("candidate_perm: {}, time: {}".format(candidate_perm, max(cost_list)))
        
        keep_searching = True
        
        while keep_searching:
            rand_perm = choice(candidate_perm)
            if debug: print("perm: {}".format(rand_perm))
            seq_new = self.permute_seq(seq, rand_perm, stops_to_add=stops_to_add, truck_to_remove=truck_to_remove)
            # Check constraints
            if max(self.cost(seq_new)) < self.time_limit:
                keep_searching = False
              
        return seq_new
    
    #@time_it
    def simulated_annealing(self):
        
        # only choose initial station for pickup* 
        seq_curr = self.gen_init_seq()
        
        obj_curr = self.objective(seq_curr)
        seq_best = seq_curr.copy()
        obj_best = obj_curr
        i = 0
        i_tol = 0
        
        # start iteration from the first position 
        while (i < self.iter_max) & (i_tol < self.tolerance):
            
            seq_new = self.gen_new_seq(seq_curr)
            obj_curr = self.objective(seq_curr)
            obj_new = self.objective(seq_new)
            
            # Update the local solution
            if obj_curr <= obj_new:# maximize？
                seq_curr = seq_new
                obj_curr = obj_new
            elif r.uniform(0, 1) < np.exp(-(obj_new - obj_curr) / (self.K + self.temp)):
                seq_curr = seq_new
                obj_curr = obj_new
            
            # Update the best solution
            if (obj_curr < obj_best) | \
                ((obj_curr == obj_best) & (max(self.cost(seq_curr)) < max(self.cost(seq_best)))):
                obj_best = obj_curr
                seq_best = seq_curr
                i_tol = 0
            else:
                i_tol += 1
            
            if (i % self.temp_schedule == 0) & (i > 0):
                self.temp *= self.alpha
                
            i += 1
            if (self.verbose == True) & ((i + 1) % 2000 == 0):
                print('{} iterations. Best obj: {}. Temperature : {}'.format(i+1, obj_best, self.temp))
            
            if self.track_progress:
                self.progress += [{'iter': i, 'seq_curr': seq_curr, 'obj_curr': obj_curr,
                                   'seq_best': seq_best, 'obj_best': obj_best,
                                   'temp': self.temp}]
        
            if self.debug:
                print("Round - {}, Temp - {}, Route - {}, Obj - {}".format(i, self.temp, seq_curr, obj_curr))
        
        # Record
        self.redist_obj = obj_best
        self.actions, self.truck_inventory, self.station_inventory = self.gen_actions(seq_best)
        self.route = seq_best
        for truck in range(self.n_truck):
            stops_w_actions = [True] + list(np.array(self.actions[truck][1:-1]) != 0) + [True]
            self.route[truck] = list(np.array(self.route[truck])[stops_w_actions])
            self.actions[truck] = list(np.array(self.actions[truck])[stops_w_actions])
            self.truck_inventory[truck] = list(np.array(self.truck_inventory[truck])[stops_w_actions])

        if self.track_progress:
            self.progress = pd.DataFrame.from_dict(self.progress)
    
    
    def output_solution(self, verbose=True):
        
        route_ = [[self.ind_to_stop[r] for r in route] for route in self.route]
        route_redist = [[self.station_inventory[r] for r in route] for route in self.route]
        
        for truck in range(self.n_truck):
            if (self.actions[truck][0] != 0) | (self.actions[truck][-1] != 0):
                if self.actions[truck][0] == 0:
                    route_redist[truck][0] = np.nan
                else:
                    route_redist[truck][-1] = np.nan
        
        sol = {}
        for truck in range(self.n_truck):
            sol[truck] = pd.DataFrame({"stop": route_[truck],
                               "action": self.actions[truck],
                               "truck_inv": self.truck_inventory[truck],
                               "actual": [self.actual_list_raw[i] for i in route_[truck]],
                               "expected": [self.expected_list_raw[i] for i in route_[truck]], 
                               "redist": route_redist[truck]
                               })
            
        redist = [self.station_inventory[self.ind_to_stop.index(x)] if x in self.ind_to_stop \
                      else self.actual_list_raw[x] for x in range(len(self.actual_list_raw))]
        
        inv = pd.DataFrame({'stop': range(len(self.actual_list_raw)),
                            'actual': self.actual_list_raw,
                           'expected': self.expected_list_raw,
                           'redist': redist,
                           'diff': abs(np.array(redist) - np.array(self.expected_list_raw))
                           })
        output = {'iterations': self.progress.shape[0],
                'unsatisfied_customers': self.redist_obj,
                'time': max(self.cost(self.route)),
                'route_df': sol,
                'station_inv_df': inv}
        
        if verbose:
            print("Iterations: {}.".format(self.progress.shape[0]))
            print("Objective(unsatisfied customers): {}".format(self.redist_obj))
            print("Time: {}".format(output['time']))
            action_type = [['+' + str(x) if x > 0 else str(x) for x in actions] for actions in self.actions]
            for truck in range(self.n_truck):
                action_seq = ' --> '.join([str(x) + ' (' + y + ')' for x, y in zip(route_[truck], action_type[truck])])
                print("\nTruck {}: {}".format(truck, action_seq))
            
            print("\nROUTE")
            for truck in range(self.n_truck):
                print("Truck {}:".format(truck))
                print(output['route_df'][truck])
            
            print("\nSTATION INVENTORY")
            print(output['station_inv_df'])
            
        return output
        

    
    def plot_convergence(self, plot_obj_curr=False):
    
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        if plot_obj_curr:
            ln1 = ax1.plot(self.progress['obj_curr'].values, color = 'blue')
        ln2 = ax1.plot(self.progress['obj_best'].values, color = 'green')
        ln3 = ax2.plot(self.progress['temp'].values, color = 'lightgrey', linestyle='dashed')
        
        if plot_obj_curr:
            ax1.legend(ln1 + ln2 + ln3, ['Objective Local', 'Objective Best', 'Temperature'])
        else:
            ax1.legend(ln2 + ln3, ['Objective Local', 'Objective Best', 'Temperature'])
                
        plt.title("SA Convergence")
        plt.xlabel("Iterations")
        ax1.set_ylabel("Reward")
        ax2.set_ylabel("Temperature")







