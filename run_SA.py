#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:41:31 2020

@author: xuanlin
"""

import sys
sys.path.append('/Users/xuanlin/Desktop/Capstone/Optimization')
import Divvy_SA as s


# ----------------- VARIABLE ----------------- #

time_mtrx = np.array([0, 3, 12, 4,
                           3, 0, 8, 9,
                           12, 8, 0, 10,
                           4, 9, 10, 0]).reshape(4,4)
time_limit = 100
truck_capacity = 10
actual_list = [0, 5, 3, 4]
expected_list = [0, 2, 7, 3]
station_capacity = [0, 8, 8, 8]
debug = False


# ------------------ RUN ALGO ------------------ #

# Set up the problem
opt = s.SA(time_mtrx = time_mtrx,
         time_limit = time_limit,
         actual_list = actual_list, 
         expected_list = expected_list,
         station_capacity = station_capacity,
         truck_capacity = truck_capacity,
         debug = debug)


opt.simulated_annealing()  # Run the optimizatino algorithm
opt.print_solution()  # Print report
opt.plot_convergence()








