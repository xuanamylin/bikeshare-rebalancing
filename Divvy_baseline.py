#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ACO by Susan

Created on Wed Feb 10 22:26:56 2020

@author: Xiangwen Sun
"""

from copy import deepcopy


class Baseline(object):

    def __init__(self,
                 travel_time,
                 demand,
                 capacity,
                 time_constraint=60,
                 pick_up_init_station=True,
                 return_init_station=True,
                 specify_final_station=True,
                 start_station=0,
                 final_station=None,
                 num_trucks=1,
                 multiple_visits=False):

        if start_station > len(travel_time):
            raise Exception("start station is out of range")
        self.start_station = start_station
        self.final_station = final_station
        self.travel_time = travel_time
        self.all_inds = range(len(travel_time))
        self.time_constraint = time_constraint
        self.demand = demand
        self.capacity = capacity
        self.num_trucks = num_trucks
        self.multiple_visits = multiple_visits
        self.pick_up_init_station = pick_up_init_station
        self.return_init_station = return_init_station
        self.specify_final_station = specify_final_station

        self.paths = None
        self.truck_inv = None
        self.satisfy_record = None
        self.demand_lst = None
        self.truck_time_travel = None
        self.truck_selected = None
        self.bikes_on_trucks = None
        self.satisfy = None

    def run(self):
        # station related variables
        demand = list(deepcopy(self.demand))
        demand_multiple_visit = list(deepcopy(self.demand))
        time_constraint = deepcopy(self.time_constraint)
        curr_time = 0
        visited = set()
        capacity = self.capacity
        visited.add(self.start_station)

        paths = [[] for _ in range(self.num_trucks)]
        truck_inv = [[] for _ in range(self.num_trucks)]
        satisfy_record = [[] for _ in range(self.num_trucks)]
        truck_time_travel = [0 for _ in range(self.num_trucks)]
        truck_selected = [i for i in range(self.num_trucks)]
        bikes_on_trucks = [0 for _ in range(self.num_trucks)]
        satisfy = [0 for _ in range(self.num_trucks)]
        prev_moves = [self.start_station for _ in range(self.num_trucks)]

        while curr_time < time_constraint:
            if len(truck_time_travel) < 1:
                break
            curr_time = truck_time_travel.pop(0)
            print('current time:', curr_time)
            if curr_time > time_constraint:
                break
            truck = truck_selected.pop(0)
            prev = prev_moves[truck]

            # calculate maximum fulfillable capacity
            cur_num_bikes = bikes_on_trucks[truck]
            if self.multiple_visits:
                high_bar = max(demand)  # 正数
                low_bar = min(demand)  # 负数
            else:
                high_bar = max(demand_multiple_visit)  # 正数
                low_bar = min(demand_multiple_visit)  # 负数

            # demand 正数为pick-up，负数为drop-off
            pick_up_max = min(high_bar, capacity - cur_num_bikes) if high_bar > 0 else 0
            drop_off_max = min(-low_bar, cur_num_bikes) if low_bar < 0 else 0


            if pick_up_max == 0 and drop_off_max == 0:
                truck_time_travel.append(next_move_time)
                truck_selected.append(truck)
                break
            elif pick_up_max >= drop_off_max:
                move = demand.index(high_bar)
                next_move_time = curr_time + self.travel_time[prev][move]
                if next_move_time < time_constraint:
                    demand[move] -= pick_up_max
                    if not self.multiple_visits:
                        demand_multiple_visit[move] = 0
                    bikes_on_trucks[truck] += pick_up_max
                    satisfy[truck] += pick_up_max
                    paths[truck].append(move)
                    truck_inv[truck].append(bikes_on_trucks[truck])
                    satisfy_record[truck].append(pick_up_max)
                    print(
                    'truck', truck, 'pick up', pick_up_max, 'bikes at station', move, ';finish time: ', next_move_time)

            else:  # pick_up_max < drop_off_max
                move = demand.index(low_bar)
                next_move_time = curr_time + self.travel_time[prev][move]
                if next_move_time < time_constraint:
                    demand[move] += drop_off_max
                    if not self.multiple_visits:
                        demand_multiple_visit[move]=0
                    bikes_on_trucks[truck] -= drop_off_max
                    satisfy[truck] += drop_off_max
                    paths[truck].append(move)
                    truck_inv[truck].append(bikes_on_trucks[truck])
                    satisfy_record[truck].append(drop_off_max)

                    print(
                    'truck', truck, 'drop_off', drop_off_max, 'bikes at station', move, ';finish time: ', next_move_time)
            if self.num_trucks == 1:
                truck_time_travel.append(next_move_time)
                truck_selected.append(truck)
            elif truck_time_travel[0] > next_move_time:
                truck_time_travel.insert(0, next_move_time)
                truck_selected.insert(0, truck)
            else:
                truck_time_travel.append(next_move_time)
                truck_selected.append(truck)
            prev_moves[truck] = move

        self.demand_lst = demand
        self.paths = paths
        self.truck_inv = truck_inv
        self.satisfy_record = satisfy_record
        self.truck_time_travel = truck_time_travel
        self.truck_selected = truck_selected
        self.bikes_on_trucks = bikes_on_trucks
        self.satisfy = satisfy

    def result(self):
        print('demand list: ', self.demand_lst)
        print('paths: ', self.paths)
        print('truck inv', self.truck_inv)
        print('satisfied customer', sum(self.satisfy))