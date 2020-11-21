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


def GridSearchOpt(params_regular, params_tuned, func, func_name):
    """
    Args:
        params_regular: regular parameters (does not tune)
        params_tuned: parameters that need to be tuned. value in a list
        func: the function variable
        func_name: function's name, must be 'sa' or 'aco'
    """
    vals = [v for v in params_tuned.values()]
    permu_params = [dict(zip(params_tuned, p)) for p in product(*vals)]
    result_lst = []
    for params_t in permu_params:
        input_params = {**params_regular, **params_t}
        opt = func(**input_params)
        if func_name =='aco':
            output = opt.run()
            satisfy = output[2]
        elif func_name =='sa':
            opt.simulated_annealing()  # Run the optimizatino algorithm
            sa_solution = opt.output_solution(verbose = False)
            satisfy = sa_solution['satisfied_customers']
        result = str(params_t) + ":" +str(satisfy) 
        result_lst.append(result)
    return result_lst