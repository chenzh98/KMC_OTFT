# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:29:38 2019

@author: Zihao Chen
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

# The scale of the simulation 200nm*30nm*200nm
t_ox = 5
t_semi = 10
len_x = 50 
len_y = 50
len_z = t_ox + t_semi

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
def vab(carrier_3d, potential_3d, a, b):
#    global t_ox
    Ea = potential_3d[a[0], a[1], a[2]]
    Eb = potential_3d[b[0], b[1], b[2]]
    if carrier_3d[b[0], b[1], b[2]] > 0:
        return 0
#    elif b[2] < t_ox:
#        return 0
    elif Eb > Ea:
        return math.exp(-10*math.sqrt((b[0]-a[0])**2+
                                          (b[1]-a[1])**2+(b[2]-a[2])**2)-
                                          q*(Eb-Ea)/(kB*T))
    else:
        return math.exp(-10*math.sqrt((b[0]-a[0])**2+
                                          (b[1]-a[1])**2+(b[2]-a[2])**2))
    #--------------------------------------------------------------------------#
    #Given a point, get the vij to all 26 directions at the point
def v_all_dirt(carrier_3d, potential_3d, x, y, z):
    v = []#v is the hopping probability
    dirtn = []#direction
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y, z]))
    dirtn.append([x-1, y, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y, z-1]))
    dirtn.append([x-1, y, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y, z+1]))
    dirtn.append([x-1, y, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y-1, z]))
    dirtn.append([x-1, y-1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y-1, z-1]))
    dirtn.append([x-1, y-1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y-1, z+1]))
    dirtn.append([x-1, y-1, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y+1, z]))
    dirtn.append([x-1, y+1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y+1, z-1]))
    dirtn.append([x-1, y+1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x-1, y+1, z+1]))
    dirtn.append([x-1, y+1, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y, z-1]))
    dirtn.append([x, y, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y, z+1]))
    dirtn.append([x, y, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                [x, y, z], [x, y-1, z]))
    dirtn.append([x, y-1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y-1, z-1]))
    dirtn.append([x, y-1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                [x, y, z], [x, y-1, z+1]))
    dirtn.append([x, y-1, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y+1, z]))
    dirtn.append([x, y+1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y+1, z-1]))
    dirtn.append([x, y+1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x, y+1, z+1]))
    dirtn.append([x, y+1, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y, z]))
    dirtn.append([x+1, y, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y, z-1]))
    dirtn.append([x+1, y, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y, z+1]))
    dirtn.append([x+1, y, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y-1, z]))
    dirtn.append([x+1, y-1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y-1, z-1]))
    dirtn.append([x+1, y-1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y-1, z+1]))
    dirtn.append([x+1, y-1, z+1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y+1, z]))
    dirtn.append([x+1, y+1, z])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y+1, z-1]))
    dirtn.append([x+1, y+1, z-1])
    v.append(vab(carrier_3d, potential_3d, 
                 [x, y, z], [x+1, y+1, z+1]))
    dirtn.append([x+1, y+1, z+1])
    return np.array(v), np.array(dirtn)
    #v is a list of probability(v_ij) hopping to nearest sites.
    #dirt is the corresponding dirction(site).
    #---------------------------------------------------------------------#
    #Broadcast potential_2d to potential_3d
def update_pot(potential_2d, potential_3d):
    i = 0
    while i < len_x:
        potential_3d[i] = potential_2d.T 
        #历史遗留问题，最开始求解矩阵的时候yz颠倒了，所以这里需要.T进行转置
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
                        q/(4*np.pi*epsilon0*epsilon_ox)/math.sqrt((i-z)**2+(j-y)**2)
                else:
                    potential[i, j] = \
                        q/(4*np.pi*epsilon0*epsilon_s)/math.sqrt((i-z)**2+(j-y)**2)
    return potential
    #----------------------------------------------------------------------#
    #for parallel computing
    #Here what we need is the minimum rt, and corresponding site ordinates
    #
def hopping_x_section(chunk_i, carrier_3d, potential_3d):
    #这个函数只处理非边界的情况，所以索引不会包括最外面的一层
    #carrier_3d = np.array(c-carrier)
    #potential_3d = np.array(c-potential)
    global t_ox
    #1000 is meaningless. Just a large enough number to start.
    #rt means resident time
    #v is the probability(vij), dirt is corresponding dirction
    #print("enter child process!!!")
    #visualize(carrier_3d)
    rt_min = 1000#1000 is meaningless. Just a large enough name to start
    x = 0
    while x < np.shape(carrier_3d)[0]:
        y = 0
        while y < np.shape(carrier_3d)[1]:
            z = t_ox 
            while z < np.shape(carrier_3d)[2]:
                if carrier_3d[x, y, z] == 1:
                    v, dirt = v_all_dirt(carrier_3d, potential_3d, x, y, z)
                    if v.sum() > 0:
                        rt_i = -math.log(random.random())/v.sum()/v0
                        if rt_i < rt_min:
                            rt_min = rt_i
                            v_hop = v
                            dirt_hop = dirt
                            hop_ini = np.array([x, y, z], dtype = int)
                z += 1
            y += 1
        x += 1
    #print("signal2!!!!!!!!!!!!@!!!!!")
    #print(hop_site)
    return rt_min, hop_site
    #print("enter child process!!!")
#_init function is for array sharing
"""
def _init(carrier, potential):
    print("initialize!!!!!!")
    global c_carrier
    global c_potential
    c_carrier = carrier
    c_potential = potential
"""
#---------------------------------------------------------------------------#
#In the end, only the minimum resident time and coresponding hopping site are 
#recorded in site_record.
#The following functions is for the callback in the pool.
def paral_site_record(rt_and_site):
    global site_record
    site_record.append(rt_and_site)
    print("!!!!!!!!!!!callback!!!!!!!!!!!!!!")
    #print(site_record)
    print("Above is site_record!!!!!!!!%d"%len(site_record))
#-------------------------------------------------------------#
if __name__=='__main__':
    #---------Initialization-------------#
    begin = time.time()
    sys_time = 0
    time_counter = 0
    set_time = 200  #set the running time, 1000 times hopping
    current_counter = 0
    #parallel computing
    cores = 4#mp.cpu_count()
    #-------------------------------------------------------------------#        
    carrier_3d = np.zeros((len_x, len_y, len_z), dtype = int)
    potential_3d = np.zeros((len_x, len_y, len_z))
    potential_2d = np.zeros((len_z, len_y))#z is the row num, y is the column num
    q_num = np.zeros((len_z, len_y))
    # define the source/drain.
    #1 means a carrier has occupied the lattice point. 0 means no carrier on the site.
    carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1 #Source side 
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
    boundary_pot(potential_2d)
    #show_mat(potential_2d)
    #visualize(carrier_3d)
    update_pot(potential_2d, potential_3d)
    np.save('potential_3d' + '_' + str(len_x) + '_' 
            + str(len_y) + '_' + str(len_z) + '.npy', potential_3d)
    #pot_record = []
#--------------------------------------------------------------------------#
    #start hopping
    while time_counter < set_time:# set the running time of the simulation
        site_record = []
        #hopping()
        #----------------------------start Hopping------------------------------------------
        """
        #Hopping change carrier_3d and system time.
        #Potential_3d won't be update in this function.
        #One charge is hopping.
        """
        #To save the time of "pickling", \
        #c-type shared Arrays are created, which can be shared between processes.
        """
        manager = Manager()
        shared_carrier = manager.list(carrier_3d.tolist())
        shared_pot = manager.list(potential_3d.tolist())
        """
        #首先对carrier_3d进行分割，放进子进程
        chunk = np.shape(carrier_3d)[0] // cores
        p = Pool(processes=cores)
        for i in range(cores):
            slice_of_carrier_3d = slice(i*chunk, 
                                        np.shape(carrier_3d)[0] if i == cores-1 else (i+1)*chunk+2)
            p.apply_async(hopping_x_section, args=(i, carrier_3d[slice_of_carrier_3d, :, 5:], 
                                                      potential_3d[slice_of_carrier_3d, :, 5:]), 
                                            callback=paral_site_record)
        p.close()
        p.join()
        print("signal 1")
        print("len of site_record: %d"%len(site_record))
        rt_min = 1000#1000 is meaningless. Just a large enough number to start.
        for rt_and_site in site_record:
            if rt_and_site[0] < rt_min:
                rt_min = rt_and_site[0]
                hop_ini = rt_and_site[1]
        #Above process finds the carrier that hops. 
        #And the probabilities to all 26 directions respectively. 
        #Yet we still need the hopping direction.
        v_hop, dirt_hop = v_all_dirt(carrier_3d, potential_3d, 
                                    hop_ini[0], hop_ini[1], hop_ini[2])
        print("v_hop and dirt_hop")
        print(v_hop)
        print(dirt_hop)
        rdm2 = random.random()
        i = 0
        while i < len(v_hop):
            if (rdm2 > v_hop[:i].sum()/v_hop.sum()) and\
            (rdm2 <= v_hop[:i+1].sum()/v_hop.sum()):
                hop_finl = np.array(dirt_hop[i], dtype = int)
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
        boundary_pot(potential_2d)
        update_pot(potential_2d, potential_3d)
        #Now consider source and drain current
        #Once a charge carrier leaves source, the empty site would be refilled.
        #Once a charge carrier reaches drain, the charge would be removed.
        if hop_finl[1] == 0:
            current_counter += 1
        carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1 #Set the boundary again
        carrier_3d[:, 0, t_ox-1:len_z-1] = 0
        current = current_counter*q/sys_time 
        if time_counter == set_time - 1:
            #pot_record.append(potential_2d)
            show_mat(potential_2d)
            visualize(carrier_3d)   
    #--------------------------------------------------------------------#
    end = time.time()
    print("Runtime: %0.4f"%(end - begin))