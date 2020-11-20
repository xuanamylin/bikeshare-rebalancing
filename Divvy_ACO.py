#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ACO by Susan

Created on Thu Nov 19 17:26:56 2020

@author: xuanlin
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
                 verbose = False):
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
        self.all_time_best_path = ("placeholder", np.inf, 0, 0, self.demand)
        self.verbose = verbose

    def run(self):
        best_path = None
        all_time_best_path = ("placeholder", np.inf, 0, 0, self.demand)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, best_path=best_path)
            best_path = max(all_paths, key = lambda x: x[2]/x[1])
            if self.verbose:
                print('---------------------iteration ',i+1,'------------------------')
                print('best_path: ', best_path[0])
                print('time traveled', best_path[1])
                print('satisfied customer', best_path[2])
                print('unsatisfied customer', sum(abs(self.demand)) - best_path[2])
                print('final vehicle inventory', best_path[3])
                print('demand left', best_path[4])
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

    def update_vehicle(self, vehicle, satisfy, move, demand):
        if vehicle + demand[move] <0:
            satisfy += vehicle
            demand[move] += vehicle
            vehicle =0
        elif vehicle + demand[move] <= self.capacity:
            satisfy += abs(demand[move])
            vehicle = vehicle + demand[move]
            demand[move] =0
        else: # vehicle + self.demand[move] > self.capacity
            satisfy += self.capacity - vehicle
            demand[move] = demand[move]- (self.capacity - vehicle)
            vehicle = self.capacity
        return satisfy, vehicle, demand

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
    
    def spread_pheromone(self, all_paths, n_best, best_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist, satisfy,__,__ in sorted_paths[:n_best]:
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
            path, satisfy, vehicle, demand = self.gen_path(self.start_station) 
            all_paths.append((path, self.gen_path_dist(path), satisfy, vehicle, demand))
        return all_paths

    
    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        demand = deepcopy(self.demand)
        remaining_travel_time = self.time_constraint
        # start inventory
        vehicle = 0
        satisfy = 0
        
        # pick up bikes from initial station  
        satisfy, vehicle, demand = self.update_vehicle(vehicle, satisfy, start, demand)
            
        for i in range(len(self.travel_time) - 1):
            move = self.pick_move(self.pheromone[prev], self.travel_time[prev], visited)
            # check if we can take this move 
            if remaining_travel_time > self.travel_time[prev][move]+self.travel_time[move][start]:
                remaining_travel_time -= self.travel_time[prev][move]
                # calculate how much demand satisfied by this move & how the vehicle would change
                satisfy, vehicle, demand = self.update_vehicle(vehicle, satisfy, move, demand)
                path.append((prev, move))
                prev = move
                visited.add(move)
            else: 
            # there is not enough time left to travel to the next station
            # break and go back to the starting point directly
                break
        path.append((prev, start)) # going back to where we started    
        remaining_travel_time -= self.travel_time[prev][start]
        
        satisfy, vehicle, demand = self.update_vehicle(vehicle, satisfy, start, demand)
        return path, satisfy, vehicle, demand
    
    





