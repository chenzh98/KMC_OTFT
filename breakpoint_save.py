# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:12:49 2019

@author: Zihao Chen
"""
import numpy as np
def savenpy(arr, name, time_counter):
    np.save(name + '_' +str(time_counter) + '.npy', arr)
savenpy(carrier_3d, "carrier_3d", time_counter)
savenpy(potential_3d, "potential_3d", time_counter)
savenpy(current_record, "current_record", time_counter)
savenpy(time_record, "time_record", time_counter)