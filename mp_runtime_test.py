# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:29:38 2019

@author: Zihao Chen
version 2 for multiprocessing
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random 
from scipy.linalg import solve
import multiprocessing as mp
from multiprocessing import Pool, sharedctypes, Manager
import time
#-----------------------------------------------------------------------------------
#define the constants
epsilon0 = 8.854e-12
epsilon_ox = 12
epsilon_s = 4
q = 1.6e-19  #elemental charge
lat_c = 1e-9   #lattice constant
kB = 1.38e-23
T = 300
v0 = 1
# The scale of the simulation 200nm*30nm*200nm
t_ox = 5
t_semi = 10
len_x = 50 
len_y = 50
len_z = t_ox + t_semi
#------------------cores for multiprocessing---------------------------------#
cores = 12#mp.cpu_count()
#----------------------------define functions---------------------------------#
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
    #calc the probability of hopping from site a to b
    #这里采用MA公式
    #Broadcast potential_2d to potential_3d
def update_pot(potential_2d, potential_3d):
    i = 0
    while i < len_x:
        potential_3d[i] = potential_2d.T 
        #历史遗留问题，最开始求解矩阵的时候yz颠倒了，所以这里需要.T进行转置
        i += 1
    #--------------------------------------------------------------------#
def set_boundary_pot(potential_2d):
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
                        q/(4*np.pi*epsilon0*epsilon_ox)/math.sqrt((i-z)**2+(j-y)**2)
                else:
                    potential[i, j] = \
                        q/(4*np.pi*epsilon0*epsilon_s)/math.sqrt((i-z)**2+(j-y)**2)
    return potential
    #----------------------------------------------------------------------#
    #for parallel computing
    #Here what we need is the minimum rt, and corresponding site ordinates
    #
def vab(carrier_3d, potential_3d, a, b):
    #为了处理边界问题，这里捕获了索引异常，但是-1任然会被索引，所以还是有三个面边界需要特殊处理
    try:
        Ea = potential_3d[a[0], a[1], a[2]]
        Eb = potential_3d[b[0], b[1], b[2]]
        if carrier_3d[b[0], b[1], b[2]] > 0:
            return 0
        elif b[2] < t_ox:
            return 0
        elif b[0] < 0 or b[1] < 0:
            return 0
        elif Eb > Ea:
            return math.exp(-10*math.sqrt((b[0]-a[0])**2+
                                              (b[1]-a[1])**2+(b[2]-a[2])**2)-
                                              q*(Eb-Ea)/(kB*T))
        else:
            return math.exp(-10*math.sqrt((b[0]-a[0])**2+
                                              (b[1]-a[1])**2+(b[2]-a[2])**2))
    except IndexError:
        return 0
    #--------------------------------------------------------------------------#
    #Given a point, get the vij to all 26 directions at the point
def v_all_drt(carrier_3d, potential_3d, x, y, z, get_drt=False):
    x_neighbor = [-1, 0, 1]
    y_neighbor = [-1, 0, 1]
    z_neighbor = [-1, 0, 1]  
    v = []#v is the hopping probability
    drtn = []#direction
    if get_drt == False:
        for i in x_neighbor:
            for j in y_neighbor:
                for k in z_neighbor:
                    v.append(vab(carrier_3d, potential_3d, 
                                 [x, y, z], [x+i, y+j, z+k]))   
    else:
        for i in x_neighbor:
            for j in y_neighbor:
                for k in z_neighbor:
                    v.append(vab(carrier_3d, potential_3d, 
                                 [x, y, z], [x+i, y+j, z+k]))
                    drtn.append([x+i, y+j, z+k])
    assert v != [], 'v is empty'
    #print(v)
    return np.array(v), np.array(drtn)
    #v is a list of probability(v_ij) hopping to nearest sites.
    #drt is the corresponding dirction(site).
    #---------------------------------------------------------------------#
def hopping_x_section(chunk_i, chunk, carrier_3d, potential_3d):
    #这个函数只处理非边界的情况，所以索引不会包括最外面的一层
    #print("enter child process!!!")
    #visualize(carrier_3d)
    rt_min = 1000#1000 is meaningless. Just a large enough name to start
    if chunk_i == cores - 1:
        for x in range(1, np.shape(carrier_3d)[0]):
            for y in range(0, np.shape(carrier_3d)[1]):
                for z in range(5, np.shape(carrier_3d)[2]):
                    if carrier_3d[x, y, z] == 1:
                        v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z)#drt在这里暂时没用
                        if v.sum() > 0:
                            #print("I did find something useful!")
                            rt_i = -math.log(random.random())/v.sum()/v0
                            if rt_i < rt_min:
                                rt_min = rt_i
                                hop_ini = np.array([x, y, z], dtype = int)
    elif chunk_i == 0:
        for x in range(0, np.shape(carrier_3d)[0]-1):
            for y in range(0, np.shape(carrier_3d)[1]):
                for z in range(5, np.shape(carrier_3d)[2]):
                    if carrier_3d[x, y, z] == 1:
                        v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z)#drt在这里暂时没用
                        if v.sum() > 0:
                            #print("I did find something useful!")
                            rt_i = -math.log(random.random())/v.sum()/v0
                            if rt_i < rt_min:
                                rt_min = rt_i
                                hop_ini = np.array([x, y, z], dtype = int)
    else:
        for x in range(1, np.shape(carrier_3d)[0]-1):
            for y in range(0, np.shape(carrier_3d)[1]):
                for z in range(5, np.shape(carrier_3d)[2]):
                    if carrier_3d[x, y, z] == 1:
                        v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z)#drt在这里暂时没用
                        if v.sum() > 0:
                            #print("I did find something useful!")
                            rt_i = -math.log(random.random())/v.sum()/v0
                            if rt_i < rt_min:
                                rt_min = rt_i
                                hop_ini = np.array([x, y, z], dtype = int)
    #print("signal2!!!!!!!!!!!!@!!!!!")
    #print(hop_site)
    hop_ini += np.array([chunk_i*chunk, 0, 0])#返回的必须是真实的坐标
    return rt_min, hop_ini#这里只是为了返回跳跃的位置和停留时间
    #print("enter child process!!!")
"""
def bottom_hopping(carrier_3d, potential_3d):
    rt_min = 1000
    z = 0
    #print('Enter bottom_hopping!!!!')
    for x in range(1, np.shape(carrier_3d)[0]):
        for y in range(1, np.shape(carrier_3d)[1]):
            if carrier_3d[x, y, z] == 1:
                v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z, boundary='bottom')
                if v.sum() > 0:
                    rt_i = -math.log(random.random())/v.sum()/v0
                    if rt_i < rt_min:
                        rt_min = rt_i
                        hop_ini = np.array([x, y, z], dtype = int)
    hop_ini += np.array([0, 0, t_ox])
    return rt_min, hop_ini   
def front_hopping(carrier_3d, potential_3d):
    rt_min = 1000
    x = 1
    for z in range(1, np.shape(carrier_3d)[2]):
        for y in range(1, np.shape(carrier_3d)[1]):
            if carrier_3d[x, y, z] == 1:
                v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z, boundary='front')
                if v.sum() > 0:
                    rt_i = -math.log(random.random())/v.sum()/v0
                    if rt_i < rt_min:
                        rt_min = rt_i
                        hop_ini = np.array([x, y, z], dtype = int)
    hop_ini += np.array([len_x-2, 0, t_ox])
    return rt_min, hop_ini 
def back_hopping(carrier_3d, potential_3d):
    #不包含y轴上的点
    rt_min = 1000
    x = 0
    for z in range(1, np.shape(carrier_3d)[2]):
        for y in range(1, np.shape(carrier_3d)[1]):
            if carrier_3d[x, y, z] == 1:
                v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z, boundary='back')
                if v.sum() > 0:
                    rt_i = -math.log(random.random())/v.sum()/v0
                    if rt_i < rt_min:
                        rt_min = rt_i
                        hop_ini = np.array([x, y, z], dtype = int)
    hop_ini += np.array([0, 0, t_ox])
    return rt_min, hop_ini 
def y_axis_hopping(carrier_3d, potential_3d):
    rt_min = 1000
    x = 0
    z = 0
    for y in range(1, np.shape(carrier_3d)[1]):
        if carrier_3d[x, y, z] == 1:
            v, drt = v_all_drt(carrier_3d, potential_3d, x, y, z, boundary='y_axis')
            if v.sum() > 0:
                rt_i = -math.log(random.random())/v.sum()/v0
                if rt_i < rt_min:
                    rt_min = rt_i
                    hop_ini = np.array([x, y, z], dtype = int) 
    hop_ini += np.array([0, 0, t_ox])
    return rt_min, hop_ini
"""
#---------------------------------------------------------------------------#
#In the end, only the minimum resident time and coresponding hopping site are 
#recorded in site_record.
#The following functions is for the callback in the pool.
def paral_site_record(rt_and_site):
    global site_record
    global rt_record
    site_record.append(rt_and_site[1])#返回的必须是真实的坐标
    rt_record.append(rt_and_site[0])
def solve_potential(len_x, len_y, len_z):
    potential_3d = np.zeros((len_x, len_y, len_z))
    potential_2d = np.zeros((len_z, len_y))#z is the row num, y is the column num
    #Vd = -10V, Vs = 0, Vg = -10V, other boundary values are 0
    #Use finite difference to solve the potential(y,z)
    #Since the boundary potential is given, we need to calc a (30-2)x(200-2) matrix
    pot_5544 = np.zeros((len_z-2)*(len_y-2))#should be 5544, means 28*198 unknown
    para_mat = np.zeros(((len_z-2)*(len_y-2), (len_z-2)*(len_y-2)))#5544, 5544
    #solve initial potential
    for i in range(0, len_y-2):
        pot_5544[i] = 10
    for j in range(t_ox-1, len_z-2, 1):
        pot_5544[j*(len_y-2)] = 10
    row = 0
    while row < np.shape(para_mat)[0]:
        dn = row // (len_y-2)
        if row % (len_y-2) == 0:
            para_mat[row, dn*(len_y-2)] = -4
            para_mat[row, dn*(len_y-2)+1] = 1        
            if dn > 0:
                para_mat[row, (dn-1)*(len_y-2)] = 1
            if dn < (len_z-3):
                para_mat[row, (dn+1)*(len_y-2)] = 1
            row += 1
        elif row % (len_y-2) == 1:
            para_mat[row, dn*(len_y-2)] = 1 
            para_mat[row, dn*(len_y-2)+1] = -4
            para_mat[row, dn*(len_y-2)+2] = 1        
            if dn > 0:
                para_mat[row, (dn-1)*(len_y-2)+1] = 1
            if dn < (len_z-3):
                para_mat[row, (dn+1)*(len_y-2)+1] = 1 
            row += 1
        elif row % (len_y-2) == (len_y-3):
            para_mat[row, dn*(len_y-2)+(len_y-4)] = 1 
            para_mat[row, dn*(len_y-2)+(len_y-3)] = -4
            if dn > 0:
                para_mat[row, (dn-1)*(len_y-2)+(len_y-3)] = 1
            if dn < (len_z-3):
                para_mat[row, (dn+1)*(len_y-2)+(len_y-3)] = 1 
            row += 1
        else:
            for i in range(0, (len_z-2)*(len_y-2)):
                para_mat[row, i] = para_mat[row-1, i-1]
            row += 1
    pot_sol = solve(para_mat, pot_5544)
    i = 1
    while i < (len_z-1):
        potential_2d[i, 1:(len_y-1)] = pot_sol[(i-1)*(len_y-2):i*(len_y-2)]
        i += 1
    set_boundary_pot(potential_2d)
    #show_mat(potential_2d)
    update_pot(potential_2d, potential_3d)
    np.save('potential_3d' + '_' + str(len_x) + '_' 
            + str(len_y) + '_' + str(len_z) + '.npy', potential_3d)
    return potential_2d, potential_3d

#-------------------------------------------------------------#
if __name__=='__main__':
    #---------Initialization-------------#
    begin = time.time()
    sys_time = 0
    time_counter = 0
    set_time = 10  #set the running time, 1000 times hopping
    current_counter = 0
    #-------------------------------------------------------------------#        
    carrier_3d = np.zeros((len_x, len_y, len_z), dtype = int)
    carrier_3d[:, len_y-1, t_ox:len_z-1] = 1
    q_num = np.zeros((len_z, len_y))
    potential_2d, potential_3d = solve_potential(len_x, len_y, len_z)
    #visualize(carrier_3d)
#--------------------------------------------------------------------------#
    #start hopping
    while time_counter < set_time:# set the running time of the simulation
        site_record = []
        rt_record = []
        hop_begin = time.time()
        #hopping()
        #----------------------------start Hopping--------------------------
        """
        #Hopping change carrier_3d and system time.
        #Potential_3d won't be update in this function.
        #One charge is hopping.
        """
        #首先对carrier_3d按x轴方向进行分割，放进子进程
        chunk = np.shape(carrier_3d)[0] // cores
        p = Pool(processes=cores)
        for i in range(cores):
            slice_of_carrier_3d = slice(i*chunk, 
                                        np.shape(carrier_3d)[0] if i == cores-1 else (i+1)*chunk+2)
            p.apply_async(hopping_x_section, args=(i, chunk, carrier_3d[slice_of_carrier_3d, :, :], 
                                                             potential_3d[slice_of_carrier_3d, :, :]), 
                                            callback=paral_site_record)
        p.close()
        p.join()
        signal_1 = time.time()
        print("signal 1 %.4f"%(signal_1 - hop_begin))
        rt_min = min(rt_record)
        hop_ini = site_record[rt_record.index(rt_min)]
        v_hop, drt_hop = v_all_drt(carrier_3d, potential_3d, 
                                    hop_ini[0], hop_ini[1], hop_ini[2], get_drt=True)
        rdm2 = random.random()
        i = 0
        while i < len(v_hop):
            if (rdm2 > v_hop[:i].sum()/v_hop.sum()) and\
            (rdm2 <= v_hop[:i+1].sum()/v_hop.sum()):
                hop_finl = np.array(drt_hop[i], dtype = int)
                break
            i += 1       
        carrier_3d[hop_ini[0], hop_ini[1], hop_ini[2]] = 0
        carrier_3d[hop_finl[0], hop_finl[1], hop_finl[2]] = 1 
    # the boundary of carrier_3d would be set again later.
        sys_time += rt_min 
        time_counter += 1
        if rt_min == 1000: 
            print("Error!")
        if time_counter % 10 == 0:
            print(time_counter)
        """
--------------------------------------------------------------------------------------------------- 
---------------------------------------------------------------------------------------------------
        """
        if (hop_finl[1]>0) and (hop_finl[2]<(len_z-1)) and (hop_finl[2]>(t_ox-1)):
            q_num[hop_finl[2], hop_finl[1]] += 1
            potential_2d += single_charge_pot(hop_finl[1], hop_finl[2])
        if (hop_ini[1]<(len_y-1)) and (hop_ini[2]<(len_z-1)):
            q_num[hop_ini[2], hop_ini[1]] -= 1
            potential_2d -= single_charge_pot(hop_ini[1], hop_ini[2])
        set_boundary_pot(potential_2d)
        update_pot(potential_2d, potential_3d)
        #Now consider source and drain current
        #Once a charge carrier leaves source, the empty site would be refilled.
        #Once a charge carrier reaches drain, the charge would be removed.
        if hop_finl[1] == 0:
            current_counter += 1
        carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1 #Set the boundary again
        carrier_3d[:, 0, t_ox-1:len_z-1] = 0
        current = current_counter*q/sys_time 
        """
        if time_counter == set_time - 1:
            #pot_record.append(potential_2d)
            show_mat(potential_2d)
            visualize(carrier_3d) 
        """  
    #--------------------------------------------------------------------#
    end = time.time()
    print("Runtime: %0.4f"%(end - begin))