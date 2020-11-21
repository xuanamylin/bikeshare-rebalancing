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
                 start_station=0, 
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
        self.multiple_visits = multiple_visits
        self.all_time_best_path = ("placeholder", np.inf, 0, 0, self.demand)

    def run(self):
        best_path = None
        all_time_best_path = ("placeholder", np.inf, 0, 0, self.demand,0,0)
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
            print('final vehicle inventory', self.all_time_best_path[3])
            print('demand left', self.all_time_best_path[4])
            print('bike pick up/ drop off action', self.all_time_best_path[5])
            print('truck inventory list', self.all_time_best_path[6])

    def update_vehicle(self, vehicle, satisfy, move, demand):
        num_bikes_moved = 0 
        #print('vehicle ',vehicle)
        #print('demand ', demand)
        #print('demand is :',demand[move])

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
        #print('bikes is:', num_bikes_moved)
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
            #print('wow')
        else:
            row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
            norm_row = row / row.sum()
            move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
    
    def spread_pheromone(self, all_paths, n_best, best_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist, satisfy,__,__,__,__ in sorted_paths[:n_best]:
            for move in path:
                # Objective: satisfy the most customers
                self.pheromone[move] += 1/ ((sum(abs(self.demand))+1-satisfy)+1)

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.travel_time[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path, satisfy, vehicle, demand, bikes, truck_inv = self.gen_path(self.start_station) 
            all_paths.append((path, self.gen_path_dist(path), satisfy, vehicle, demand, bikes, truck_inv))
        return all_paths

    
    def gen_path(self, start):
        path = []
        bikes_moved =[]
        truck_inv =[]
        visited = set()
        visited.add(start)
        prev = start
        demand = deepcopy(self.demand)
        remaining_travel_time = self.time_constraint
        # start inventory
        vehicle = 0
        satisfy = 0

        print("vehicle ",vehicle)
        print('demand', demand[start])
        print('demand lst ',demand)
        
        # pick up bikes from initial station  
        satisfy, vehicle, demand, bikes = self.update_vehicle(vehicle, satisfy, start, demand)
        print('vehicle:gp ',vehicle)
        print('bikes:gp ', bikes)
        truck_inv.append(vehicle)
        bikes_moved.append(bikes)
            
        for i in range(len(self.travel_time) - 1):
            move = self.pick_move(self.pheromone[prev], self.travel_time[prev], visited, vehicle,demand)
            # check if we can take this move 
            if move ==-1:
                break
            elif remaining_travel_time > self.travel_time[prev][move]+self.travel_time[move][start]:
                remaining_travel_time -= self.travel_time[prev][move]
                # calculate how much demand satisfied by this move & how the vehicle would change
                satisfy, vehicle, demand,bikes = self.update_vehicle(vehicle, satisfy, move, demand)
                #print('demand lst: ', demand)
                path.append((prev, move))
                prev = move
                visited.add(move)
                bikes_moved.append(bikes)
                truck_inv.append(vehicle)
            else: 
            # there is not enough time left to travel to the next station
            # break and go back to the starting point directly
                break
        path.append((prev, start)) # going back to where we started    
        remaining_travel_time -= self.travel_time[prev][start]
        
        satisfy, vehicle, demand,bikes = self.update_vehicle(vehicle, satisfy, start, demand)
        bikes_moved.append(bikes)
        truck_inv.append(vehicle)
        return path, satisfy, vehicle, demand, bikes_moved, truck_inv
    

