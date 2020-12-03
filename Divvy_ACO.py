#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ACO by Susan

Created on Thu Nov 19 17:26:56 2020

@author: Xiangwen Sun
"""

import pandas as pd
import numpy as np
import random as rn
from numpy.random import choice as np_choice
from copy import deepcopy
from queue import Queue

class Ant_Colony(object):

    def __init__(self, 
                 travel_time, 
                 n_ants, 
                 n_best, 
                 n_iterations, 
                 decay, 
                 demand,
                 capacity, 
                 time_constraint = 60,
                 #start_together = True,
                 pick_up_init_station = True,
                 return_init_station = True,
                 start_station=0, 
                 final_station = None,
                 num_truck = 1,
                 alpha=0.5, 
                 beta=1,
                 multiple_visits = False):
        """
        Args:
            travel_time (2D numpy.array): Square matrix of travel_time. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Pheromone decay rate. 
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
            n_iterations: number of iterations run before it concludes on the best route
            demand(1d numpy.array): number of bikes demand at the dock, negative is demand, positive is surplus(overload)
            capacity(int): truck capacity
            multiple_visits(bool): does the algorithms allows for multiple visits to a single station
            ant_colony = Ant_Colony(time_matrix, 100, 20, 2000, 0.95, demand = demand, alpha=1, beta=2)          
        """
        if start_station > len(travel_time):
            raise Exception("start station is out of range")
        self.start_station = start_station
        self.final_station = final_station
        self.travel_time  = travel_time
        self.pheromone = np.ones(self.travel_time.shape) / len(travel_time)
        self.all_inds = range(len(travel_time))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.time_constraint = time_constraint
        self.demand = demand
        self.capacity = capacity
        self.num_truck = num_truck
        self.multiple_visits = multiple_visits
        self.pick_up_init_station = pick_up_init_station
        self.return_init_station = return_init_station
        #self.start_together = start_together 
        self.all_time_best_path = ("placeholder", np.inf,[0], [0], [0], self.demand,[0],[0])

    def run(self):
        best_path = None
        all_time_best_path = ("placeholder", np.inf,[0], [0], [0], self.demand,[0],[0])
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, best_path=best_path)
            best_path = max(all_paths, key = lambda x: x[2])
            #print('---------------------iteration ',i+1,'------------------------')
            #print('best_path: ', best_path[0])
            #print('time traveled', best_path[1])
            #print('satisfied customer', best_path[2])
            #print('unsatisfied customer', sum(abs(self.demand)) - best_path[2])
            #print('final vehicle inventory', best_path[3])
            #print('demand left', best_path[4])
            #print('bike pick up/ drop off action', best_path[5])
            #print('truck inventory list', best_path[6])
            if best_path[2] > all_time_best_path[2]:
                all_time_best_path = best_path  
            elif best_path[2] == all_time_best_path[2]:
                if best_path[1] < all_time_best_path[1]:
                    all_time_best_path = best_path  
            
            self.pheromone *= self.decay     
            self.all_time_best_path = all_time_best_path
        return all_time_best_path
    
    def best_path(self):
            print('---------------------all time best path------------------------')
            print('best_path: ', self.all_time_best_path[0])
            print('time traveled', self.all_time_best_path[1])
            print('satisfied customer', self.all_time_best_path[2])
            print('unsatisfied customer', sum(abs(self.demand)) - self.all_time_best_path[2])
            print('final vehicle inventory', self.all_time_best_path[4])
            print('demand left', self.all_time_best_path[5])
            print('bike pick up/ drop off action', self.all_time_best_path[6])
            print('truck inventory list', self.all_time_best_path[7])

    def update_vehicle(self, vehicle, satisfy, move, demand):
        num_bikes_moved = 0 

        if vehicle + demand[move] <0:
            satisfy += vehicle
            num_bikes_moved = -vehicle
            demand[move] += vehicle
            vehicle =0
        elif vehicle + demand[move] <= self.capacity:
            satisfy += abs(demand[move])
            num_bikes_moved = demand[move]
            vehicle = vehicle + demand[move]
            demand[move] =0
         
        else: # vehicle + self.demand[move] > self.capacity
            satisfy += self.capacity - vehicle
            num_bikes_moved = self.capacity - vehicle
            demand[move] = demand[move]- (self.capacity - vehicle)
            vehicle = self.capacity
        return satisfy, vehicle, demand, num_bikes_moved

    def pick_move(self, pheromone1, dist, visited, vehicle,demand):
        pheromone = np.copy(pheromone1)
        # don't visit the position that were visited before
        if not self.multiple_visits:
            pheromone[list(visited)] = 0
        
        demand_arry = np.array(demand)
        no_demand_lst = demand_arry ==0
        no_demand_idx = np.where(no_demand_lst)[0]
        # don't visit the position with no demand
        pheromone[no_demand_idx] = 0
        
        # Determine the next move by the truck capacity
        if vehicle == 0:
            no_pick_up_station_lst = demand_arry<0
            no_pick_up_station_idx = np.where(no_pick_up_station_lst)[0]
            pheromone[no_pick_up_station_idx] =0
        if vehicle == self.capacity:
            no_drop_off_station_lst = demand_arry>0
            no_drop_off_station_idx = np.where(no_drop_off_station_lst)[0]
            pheromone[no_drop_off_station_idx] =0
        
        # if no more demand could be satisfied, truck back to origin
        if sum(pheromone) == 0:
            move =-1
        else:
            row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
            norm_row = row / row.sum()
            move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
    
    def spread_pheromone(self, all_paths, n_best, best_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[2]) # sorted by total satisfy customer
        for paths, dist, satisfy,__,__,__,__,__ in sorted_paths[:n_best]:
            for path in paths:
                for move in path:
                    # Objective: satisfy the most customers
                    self.pheromone[move] += 1/ ((sum(abs(self.demand))+1-satisfy)+1)

    def gen_path_dist(self, paths):
        total_dist = 0
        for path in paths:
            dist =0
            for ele in path:
                dist += self.travel_time[ele]
                total_dist = max(dist,total_dist)
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path, satisfy, vehicle, demand, bikes, truck_inv = self.gen_path(self.start_station) 
            total_number_satisfy = np.array(satisfy).sum()
            all_paths.append((path, self.gen_path_dist(path), total_number_satisfy, satisfy, vehicle, demand, bikes, truck_inv))
        return all_paths

    
    def gen_path(self, start):
        """
        Args:
            travel_time (2D numpy.array): Square matrix of travel_time. Diagonal is assumed to be np.inf.
            paths(list of list): path of different trucks
            prev(list): list of previous stations for different trucks
            bikes_on_trucks(list): the current number of bikes in different trucks 
            satisfy(list): the current satisfied customer by each truck
            truck_inv(list of list): a list to record the truck inventory after each action for all trucks 
            bikes_moved(list of list): number of bikes moved after each action for all trucks
        Other used Args:
            self.num_truck
            self.pick_up_init_station: whether to pick up at the initial station
            self.start_together: whether all trucks are starting at the same location
        """
        # action related variables
        paths = [[] for _ in range(self.num_truck)]
        truck_inv =[[] for _ in range(self.num_truck)]
        bikes_moved = [[] for _ in range(self.num_truck)]
        # truck related variables
        prev = [start for _ in range(self.num_truck)] # if self.start_together == True
        bikes_on_trucks = [0 for _ in range(self.num_truck)]
        satisfy = [0 for _ in range(self.num_truck)]
        #print('paths: ', paths)
        # station related variables
        visited = set()
        visited.add(start)
        demand = deepcopy(self.demand)
        time_constraint = deepcopy(self.time_constraint)
        
        if self.return_init_station:
            final = start
        else:
            final = self.final_station
        
        # 在第一个station pick up
        if self.pick_up_init_station:
            for truck in range(self.num_truck):
                satisfy_on_this_truck, bikes_on_this_truck = satisfy[truck], bikes_on_trucks[truck]
                satisfy_on_this_truck, bikes_on_this_truck, demand, bikes = self.update_vehicle(bikes_on_this_truck, 
                                                                                                satisfy_on_this_truck, 
                                                                                                start, 
                                                                                                demand)
                satisfy[truck], bikes_on_trucks[truck] = satisfy_on_this_truck, bikes_on_this_truck
                truck_inv[truck].append(bikes_on_this_truck)
                bikes_moved[truck].append(bikes)

                
        # intialize current time, truck_travel_time queue, and truck selected queue.
        curr_time = 0
        
        truck_time_travel = [0 for _ in range(self.num_truck)]
        truck_selected = [i for i in range(self.num_truck)]
        
        while curr_time < time_constraint:
            if len(truck_time_travel) <1:
                break
            curr_time = truck_time_travel.pop(0)
            truck = truck_selected.pop(0)
            #print('curr_time: ', curr_time)
            #print('truck: ',truck)
            #print('ttt',truck_selected)
            
            this_truck_prev, bikes_on_this_truck = prev[truck], bikes_on_trucks[truck]
            move = self.pick_move(self.pheromone[this_truck_prev], 
                                  self.travel_time[this_truck_prev], 
                                  visited, 
                                  bikes_on_this_truck, 
                                  demand)
            truck_travel_duration = self.travel_time[this_truck_prev][move]
            #print('move',move)
            if move ==-1: # end action
                paths[truck].append((this_truck_prev, final)) # going back to where we started
                satisfy_on_this_truck, bikes_on_this_truck = satisfy[truck], bikes_on_trucks[truck]
                satisfy_on_this_truck, bikes_on_this_truck, demand, bikes = self.update_vehicle(bikes_on_this_truck, 
                                                                                                satisfy_on_this_truck, 
                                                                                                final, 
                                                                                                demand)
                satisfy[truck], bikes_on_trucks[truck] = satisfy_on_this_truck, bikes_on_this_truck
                truck_inv[truck].append(bikes_on_this_truck)
                bikes_moved[truck].append(bikes)
                prev[truck] = final
            # If there is still time to travel to the next station
            elif time_constraint - curr_time > self.travel_time[this_truck_prev][move]+self.travel_time[move][final]:
                satisfy_on_this_truck, bikes_on_this_truck = satisfy[truck], bikes_on_trucks[truck]
                satisfy_on_this_truck, bikes_on_this_truck, demand, bikes = self.update_vehicle(bikes_on_this_truck, 
                                                                                                satisfy_on_this_truck, 
                                                                                                move, 
                                                                                                demand)
                satisfy[truck], bikes_on_trucks[truck] = satisfy_on_this_truck, bikes_on_this_truck
                truck_inv[truck].append(bikes_on_this_truck)
                bikes_moved[truck].append(bikes)
                paths[truck].append((this_truck_prev, move))
                prev[truck] = move
                visited.add(move)
            # this truck go back to the intial station
            else:
                paths[truck].append((this_truck_prev, final)) # going back to where we started
                satisfy_on_this_truck, bikes_on_this_truck = satisfy[truck], bikes_on_trucks[truck]
                satisfy_on_this_truck, bikes_on_this_truck, demand, bikes = self.update_vehicle(bikes_on_this_truck, 
                                                                                                satisfy_on_this_truck, 
                                                                                                final, 
                                                                                                demand)
                #试一下 don't actually need it
                satisfy[truck], bikes_on_trucks[truck] = satisfy_on_this_truck, bikes_on_this_truck
                truck_inv[truck].append(bikes_on_this_truck)
                bikes_moved[truck].append(bikes)
                prev[truck] = final
            
            if prev[truck] != final:
                next_travel_time = truck_travel_duration+curr_time
                for i in range(len(truck_time_travel)):
                    if truck_time_travel[i] > next_travel_time:
                        truck_time_travel.insert(i,next_travel_time)
                        truck_selected.insert(i,truck)
                if truck_time_travel[-1] < next_travel_time:
                    truck_time_travel.insert(len(truck_time_travel)+1,next_travel_time)
                    truck_selected.insert(len(truck_selected)+1,truck)
                #print('final',truck_time_travel)
                #print(truck_selected)

        return paths, satisfy, bikes_on_trucks, demand, bikes_moved, truck_inv
    