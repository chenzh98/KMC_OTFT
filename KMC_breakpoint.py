# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:39:22 2019

@author: Zihao Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
#import moviepy as mvp
import random 
#from mayavi import mlab
from scipy.linalg import solve
import datetime
begin = datetime.datetime.now()
current_counter = 0
sys_time = 0
epsilon0 = 8.854e-12
epsilon_ox = 12
epsilon_s = 4
q = 1.6e-19  #elemental charge
lat_c = 1e-9   #lattice constant
kB = 1.38e-23
T = 300
Pi = 3.1415927
# The scale of the simulation 
t_ox = 5
t_semi = 10
len_x = 50 
len_y = 50
len_z = t_ox + t_semi
#------------------------------------------------------------#
def visualize(c_3d):
    x = []
    y = []
    z = []
    i = 0
    while i < len(c_3d):
        j = 0
        while j < len(c_3d[0]):
            k = 0
            while k < len(c_3d[0,0]):
                if c_3d[i, j, k] == 1:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                k += 1
            j += 1
        i += 1
    ax = plt.subplot(111, projection='3d')# 创建一个三维的绘图工程
    ax.scatter(np.array(x), np.array(y), np.array(z), c='r')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
#    plt.savefig('carrier'+ str(time_counter) + '.png')
def show_mat(Z):
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower')
    plt.show()
#    plt.savefig('potential'+ str(time_counter) + '.png')
def savenpy(arr, name):
    global time_counter
    np.save(name + '_' +str(time_counter) + '.npy', arr)
#--------------------------------------------------------------------#
#calc the probability of hopping from a to b
def vab(carrier_3d, potential_3d, a, b):
    Ea = potential_3d[a[0], a[1], a[2]]
    Eb = potential_3d[b[0], b[1], b[2]]
    if carrier_3d[b[0], b[1], b[2]] > 0:
        return 0
    elif b[2] < t_ox - 1:
        return 0
    elif Eb > Ea:
        return math.exp(-10*math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)-
                   q*(Eb-Ea)/(kB*T))
    else:
        return math.exp(-10*math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2))
#--------------------------------------------------------------------------#
#Given a point, get the vij to all 26 directions at the point
def v_all_dirt(carrier_3d, potential_3d, x, y, z):
    v = []
    dirtn = []
    if x-1 >= 0:
        v.append(vab(carrier_3d, potential_3d, 
                     [x, y, z], [x-1, y, z]))
        dirtn.append([x-1, y, z])
        if z-1 > t_ox - 1:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x-1, y, z-1]))
            dirtn.append([x-1, y, z-1])
        if z+1 < len_z:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x-1, y, z+1]))
            dirtn.append([x-1, y, z+1])
        if y-1 >= 0:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x-1, y-1, z]))
            dirtn.append([x-1, y-1, z])
            if z-1 > t_ox - 1:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x-1, y-1, z-1]))
                dirtn.append([x-1, y-1, z-1])
            if z+1 < len_z:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x-1, y-1, z+1]))
                dirtn.append([x-1, y-1, z+1])
        if y+1 < len_y:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x-1, y+1, z]))
            dirtn.append([x-1, y+1, z])
            if z-1 > t_ox - 1:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x-1, y+1, z-1]))
                dirtn.append([x-1, y+1, z-1])
            if z+1 < len_z:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x-1, y+1, z+1]))
                dirtn.append([x-1, y+1, z+1])
  #########################################################
    if z-1 > t_ox - 1:
        v.append(vab(carrier_3d, potential_3d, 
                     [x, y, z], [x, y, z-1]))
        dirtn.append([x, y, z-1])
    if z+1 < len_z:
        v.append(vab(carrier_3d, potential_3d, 
                     [x, y, z], [x, y, z+1]))
        dirtn.append([x, y, z+1])
    if y-1 >= 0:
        v.append(vab(carrier_3d, potential_3d, 
                    [x, y, z], [x, y-1, z]))
        dirtn.append([x, y-1, z])
        if z-1 > t_ox - 1:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x, y-1, z-1]))
            dirtn.append([x, y-1, z-1])
        if z+1 < len_z:
            v.append(vab(carrier_3d, potential_3d, 
                        [x, y, z], [x, y-1, z+1]))
            dirtn.append([x, y-1, z+1])
    if y+1 < len_y:
        v.append(vab(carrier_3d, potential_3d, 
                     [x, y, z], [x, y+1, z]))
        dirtn.append([x, y+1, z])
        if z-1 > t_ox - 1:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x, y+1, z-1]))
            dirtn.append([x, y+1, z-1])
        if z+1 < len_z:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x, y+1, z+1]))
            dirtn.append([x, y+1, z+1])
  ##################################################
    if x+1 < len_x:
        v.append(vab(carrier_3d, potential_3d, 
                     [x, y, z], [x+1, y, z]))
        dirtn.append([x+1, y, z])
        if z-1 > t_ox - 1:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x+1, y, z-1]))
            dirtn.append([x+1, y, z-1])
        if z+1 < len_z:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x+1, y, z+1]))
            dirtn.append([x+1, y, z+1])
        if y-1 >= 0:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x+1, y-1, z]))
            dirtn.append([x+1, y-1, z])
            if z-1 > t_ox - 1:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x+1, y-1, z-1]))
                dirtn.append([x+1, y-1, z-1])
            if z+1 < len_z:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x+1, y-1, z+1]))
                dirtn.append([x+1, y-1, z+1])
        if y+1 < len_y:
            v.append(vab(carrier_3d, potential_3d, 
                         [x, y, z], [x+1, y+1, z]))
            dirtn.append([x+1, y+1, z])
            if z-1 > t_ox - 1:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x+1, y+1, z-1]))
                dirtn.append([x+1, y+1, z-1])
            if z+1 < len_z:
                v.append(vab(carrier_3d, potential_3d, 
                             [x, y, z], [x+1, y+1, z+1]))
                dirtn.append([x+1, y+1, z+1])
    return np.array(v), np.array(dirtn)
#---------------------------------------------------------------------#
#Broadcast potential_2d to potential_3d
def update_pot(potential_2d, potential_3d):
    i = 0
    while i < len_x:
        potential_3d[i] = potential_2d.T #cuz I get the wrong y and z below
        i += 1
#--------------------------------------------------------------------#
def boundary_pot(potential_2d):
    potential_2d[0] = -10
    potential_2d[t_ox:len_z, 0] = -10
    potential_2d[t_ox:len_z, len_y-1] = 0   
#---------------------------------------------------------------------#
def single_charge_pot(y, z):
    global len_z, len_y, q
    potential = np.zeros((len_z, len_y))
    for i in range(0, len_z):
        for j in range(0, len_y):
            if i == z and j == y:
                potential[i, j] = 0
            else:
                if i <= 4:
                    potential[i, j] = \
                    q/(4*Pi*epsilon0*epsilon_ox)/math.sqrt((i-z)**2+(j-y)**2)
                else:
                    potential[i, j] = \
                    q/(4*Pi*epsilon0*epsilon_s)/math.sqrt((i-z)**2+(j-y)**2)
    return potential
#----------------------------------------------------------------------#
#Hopping change carrier_3d and system time.
#Potential_3d won't be update in this function.
#One charge is hopping.
def hopping(carrier_3d, potential_3d):  
    global sys_time
    global time_counter
    global hop_ini
    global hop_finl
    rt_min = 1000
    x = 0
    while x < np.shape(carrier_3d)[0]:
        y = 0
        while y < np.shape(carrier_3d)[1]:
            z = t_ox 
            while z < np.shape(carrier_3d)[2]:
                if carrier_3d[x, y, z] == 1:
                    v, dirt = v_all_dirt(carrier_3d, potential_3d, x, y, z)
                    if v.sum() > 0:
                        rt_i = -math.log(random.random())/v.sum()
                        if rt_i < rt_min:
                            rt_min = rt_i
                            v_hop = v
                            dirt_hop = dirt
                            hop_ini = np.array([x, y, z], dtype = int)
                z += 1
            y += 1
        x += 1
    #Above loop finds the carrier that hops. 
    #Yet we still need the hopping direction.
    rdm2 = random.random()
    i = 0
    while i < len(v_hop):
        if (rdm2 > v_hop[:i].sum()/v_hop.sum())&\
            (rdm2 <= v_hop[:i+1].sum()/v_hop.sum()):
                hop_finl = np.array(dirt_hop[i], dtype = int)
                break
        i += 1       
    carrier_3d[hop_ini[0], hop_ini[1], hop_ini[2]] = 0
    carrier_3d[hop_finl[0], hop_finl[1], hop_finl[2]] = 1 
    
    sys_time += rt_min 
    time_counter += 1
    if rt_min == 1000: 
        print("Error!")
    if time_counter % 10 == 0:
        print(time_counter)  
#-------------------------------------------------------------#
set_time = int(input('Please input set_time:'))
time_counter = int(input('Please input time_counter:'))
carrier_3d = np.load("E:\py_work\KMC_data\carrier_3d_" + \
                     str(time_counter) + ".npy")
potential_3d = np.load("E:\py_work\KMC_data\potential_3d_" + \
                       str(time_counter) + ".npy")
potential_2d = np.load("E:\py_work\KMC_data\potential_2d_" + \
                       str(time_counter) + ".npy")
q_num = np.load("E:\py_work\KMC_data\q_num_" + str(time_counter) + ".npy")
pot_record = []
#start hopping
while time_counter <= set_time:# set the running time of the simulation
    hopping(carrier_3d, potential_3d) 
    if (hop_finl[1]>0) and (hop_finl[2]<(len_z-1)) and (hop_finl[2]>(t_ox-2)):
        q_num[hop_finl[2], hop_finl[1]] += 1
        potential_2d += single_charge_pot(hop_finl[1], hop_finl[2])
    if (hop_ini[1]<(len_y-1)) and (hop_ini[2]<(len_z-1)):
        q_num[hop_ini[2], hop_ini[1]] -= 1
        potential_2d -= single_charge_pot(hop_ini[1], hop_ini[2])
    boundary_pot(potential_2d)
    update_pot(potential_2d, potential_3d)
    #Now consider source and drain current
    #Once a charge carrier leaves source, the empty site would be refilled.
    #Once a charge carrier reaches drain, the charge would be removed.
    if hop_finl[1] == 0:
        current_counter += 1
    carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1
    carrier_3d[:, 0, t_ox-1:len_z-1] = 0
    current = current_counter*q/sys_time 
    if time_counter == set_time//10 \
    or time_counter == set_time//5 \
    or time_counter == set_time//2 \
    or time_counter == set_time - 1:
        pot_record.append(potential_2d)
        show_mat(potential_2d)
        plt.savefig('potential'+ str(time_counter) + '.png')
        visualize(carrier_3d)
#--------------------------------------------------------------------#
end = datetime.datetime.now()
print(end - begin)