#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ACO by Susan

Created on Thu Nov 20 22:46:26 2020

@author: XiangwenSun
"""

import Divvy_SA as sa
import Divvy_ACO as aco
from itertools import product
import pandas as pd
import numpy as np
from copy import deepcopy 

class optimization_hypterparameter_tuning():
    def __init__(self, params_regular, params_tuned, func_name, expect_lst, actual_lst, capacity_lst = None): # 
        """
        Args:
            params_regular: regular parameters (does not tune)
            params_tuned: parameters that need to be tuned. value in a list
            func: the function variable
            func_name: function's name, must be 'sa' or 'aco'
            expect_lst: list of list of expected value in different station 
            actual_lst: list of list of actual value in different stations; length should be the same with expect_lst
        """
        self.params_regular = params_regular
        self.params_tuned = params_tuned 
        self.func_name = func_name
        self.expect_lst = np.array(expect_lst)
        self.actual_lst = np.array(actual_lst)
        self.capacity_lst = capacity_lst
        self.demand_lst = [np.array(actual_lst[i]) - np.array(expect_lst[i]) for i in range(len(actual_lst))]
        self.result = None
        
        # iterature_through different cases 
        permu_params = self.__permu_dict(self.params_tuned)
        self.result = self.__case_iteration(permu_params, self.params_regular, self.actual_lst, self.expect_lst, self.capacity_lst)
        self.report = self.__create_report(self.result)
        
        
    def __create_report(self, result):
        header = list(result[0].keys())
        answer = []
        for i in result:
            answer.append(list(i.values()))
        dataframe = pd.DataFrame(answer, columns =header)
        return dataframe
        
    def __case_iteration(self, permu_params, params_regular, actual_lst, expect_lst, capacity_lst):
        result =[]
        for i in range(len(actual_lst)):
            actual = actual_lst[i]
            expect = expect_lst[i]
            if self.func_name =='aco':
                case_result = self.__para_search_aco(permu_params, params_regular, actual, expect, i)
            elif self.func_name =='sa':
                capacity = capacity_lst[i]
                case_result = self.__para_search_sa(permu_params, params_regular, actual, expect, capacity, i)
            result.extend(case_result)
        return result


    def __satisfiable_demand(self, actual, expect):
        demand = np.array(actual) - np.array(expect)
        pick_up = sum(x for x in demand if x > 0)
        drop_off = abs(sum(i for i in demand if i < 0))
        satisfiable_demand = min(pick_up, drop_off)*2
        return satisfiable_demand

    
    def __para_search_aco(self, permu_params, params_regular, actual, expect, case):
        """
        actual and expect should be np.array
        """
        demand = np.array(actual) - np.array(expect)
        case_param = {}
        case_param['demand'] = demand
        result = []
        satisfiable_demand = self.__satisfiable_demand(actual, expect)
        #print(satisfiable_demand)
        for params_t in permu_params:
            input_params = {**case_param, **params_regular, **params_t}
            opt = aco.Ant_Colony(**input_params)
            output = opt.run()
            satisfy = output[2]
            final_params = deepcopy(params_t)
            final_params['satisfy'] = satisfy
            final_params['case'] = case
            final_params['satisfiable_demand'] = satisfiable_demand
            final_params['satisfy/satisfiable'] = satisfy / satisfiable_demand
            result.append(final_params)
        return result
                
            
    def __para_search_sa(self, permu_params, params_regular, actual, expect, capacity, case):
        """
        actual and expect should be np.array
        """
        case_param = {}
        case_param['actual_list'] = actual
        case_param['expected_list'] = expect
        case_param["station_capacity"] = capacity
        satisfiable_demand = self.__satisfiable_demand(actual, expect)
        result = []
        for params_t in permu_params:
            input_params = {**case_param, **params_regular, **params_t}
            opt = sa.SA(**input_params)
            opt.simulated_annealing() 
            sa_solution = opt.output_solution(verbose = False)
            satisfy = sa_solution['satisfied_customers']
            final_params = deepcopy(params_t)
            final_params['satisfy'] = satisfy
            final_params['case'] = case
            #final_params['satisfiable_demand'] = satisfiable_demand
            result.append(final_params)
        return result
        
    def __permu_dict(self, params_tuned):
        vals = [v for v in params_tuned.values()]
        permu_params = [dict(zip(params_tuned, p)) for p in product(*vals)]
        return permu_params

    def final_result(self):
        return self.result
    
    def compare_own_trend(self, variable, plot=None):
        if 'satisfy/satisfiable' in self.report.columns:
            if plot == 'bar':
                print(self.report.groupby(by=[variable]).mean()['satisfy/satisfiable'].plot.bar())
            else:
                print(self.report.groupby(by=[variable]).mean()['satisfy/satisfiable'].plot())
        else:
            if plot == 'bar':
                print(self.report.groupby(by=[variable]).sum()['satisfy'].plot.bar())
            else:
                print(self.report.groupby(by=[variable]).sum()['satisfy'].plot())
        
    
    def return_best(self):
        if self.func_name =='aco':
            cols = self.report.columns[-3]
        elif self.func_name =='sa':
            cols = self.report.columns[-2]
        if 'satisfy/satisfiable' in self.report.columns:
            print(self.report.groupby(by=cols).mean().sort_values(by='satisfy/satisfiable', ascending = False).head(5))
        else:
            print(self.report.groupby(by=cols).sum().sort_values(by='satisfy', ascending = False).head(5))