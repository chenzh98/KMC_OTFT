# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:12:49 2019

@author: Zihao Chen
"""
import numpy as np
def savenpy(arr, name):
    global time_counter
    np.save(name + '_' +str(time_counter) + '.npy', arr)
savenpy(carrier_3d, "carrier_3d")
savenpy(potential_3d, "potential_3d")
savenpy(potential_2d, "potential_2d" )
savenpy(q_num, "q_num")