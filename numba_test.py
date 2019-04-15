# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
#import moviepy as mvp
import random 
#from mayavi import mlab
from scipy.linalg import solve
from numba import njit
import time
data_path = 'D:\\pywork\\KMC\\data'
begin = time.time()
sys_time = 0
time_counter = 0
set_time = 50 #set the running time, 1000 times hopping
current_counter = 0
#-------------------------------------------------------------#
#define the constants
epsilon0 = 8.854e-12
epsilon_ox = 12
epsilon_s = 4
q = 1.6e-19  #elemental charge
lat_c = 1e-9   #lattice constant
kB = 1.38e-23
T = 300
#Pi = 3.1415927
v0 = 1e12
# The scale of the simulation 200nm*30nm*200nm
t_ox = 5
t_semi = 10
len_x = 50 
len_y = 50
len_z = t_ox + t_semi
vg = -10
vd = -10
#-------------------define functions-------------------------#
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
    plt.colorbar(im, shrink=0.45)
    plt.show()
#    plt.savefig('potential'+ str(time_counter) + '.png')
def savenpy(arr, name, time_counter):
    np.save(name + '_' +str(time_counter) + '.npy', arr)
#--------------------------------------------------------------------#
#Broadcast potential_2d to potential_3d
@njit
def update_pot(potential_2d, potential_3d):
    for i in range(len_x):
        potential_3d[i] = potential_2d.T #cuz I mix z and y when calculating potential_2d
    return potential_3d
#--------------------------------------------------------------------#
def boundary_pot(potential_2d):
    potential_2d[0] = -10
    potential_2d[t_ox:len_z, 0] = -10
    potential_2d[t_ox:len_z, len_y-1] = 0   
    return potential_2d
#---------------------------------------------------------------------#
@njit
def single_charge_pot(x, y, z):
    potential = np.zeros((len_x, len_y, len_z))
    for i in range(0, len_x):
        for j in range(1, len_y-1):
            for k in range(0, len_z):
                if i == x and j == y and k == z:
                    potential[i, j, k] = 0
                else:
                    if k < t_ox :
                        potential[i, j, k] = \
                        q/(4*np.pi*epsilon0*epsilon_ox)/math.sqrt((i-x)**2+(j-y)**2+(k-z)**2)
                    else:
                        potential[i, j, k] = \
                        q/(4*np.pi*epsilon0*epsilon_s)/math.sqrt((i-x)**2+(j-y)**2+(k-z)**2)
    return potential
#----------------------------------------------------------------------#
#Hopping change carrier_3d and system time.
#Potential_3d won't be update in this function.
#One charge is hopping.
@njit
def hopping(sys_time, carrier_3d, potential_3d, constants):
    (len_x, len_y, len_z, t_ox) = constants 
    rt_min = 1000#1000 is meaningless. Just a large enough name to start
    x_neighbor = [-1, 0, 1]
    y_neighbor = [-1, 0, 1]
    z_neighbor = [-1, 0, 1] 
    for x in range(len_x):
        for y in range(len_y):
            for z in range(t_ox, len_z):
                if carrier_3d[x, y, z] == 1:
                    #---------------------v_all_drt-------------------------------------------------------#
                    v = np.zeros(27)#v is the hopping probability
                    drtn = np.zeros((27, 3))#direction
                    counter = 0
                    for i in x_neighbor:
                        for j in y_neighbor:
                            for k in z_neighbor:
                                """
                                a = [x, y ,z]
                                b = [x+i, y+j, z+k]
                                """
                                if x+i < len_x and y+j < len_y and z+k < len_z:
                                    Ea = potential_3d[x, y ,z]
                                    Eb = potential_3d[x+i, y+j, z+k]
                                    if carrier_3d[x+i, y+j, z+k] > 0:
                                        vab = 0
                                    elif z+k < t_ox:
                                        vab = 0
                                    elif x+i < 0 or y+j < 0:
                                        vab = 0
                                    elif Eb > Ea:
                                        vab = math.exp(-10*math.sqrt(i**2+j**2+k**2)-
                                              q*(Eb-Ea)/(kB*T))
                                    else:
                                        vab = math.exp(-10*math.sqrt(i**2+j**2+k**2))
                                else:
                                    vab = 0
                                v[counter] = vab
                                drtn[counter] = np.array([x+i, y+j, z+k])
                                counter += 1
                    #-------------------------------------------------------------------------------------#
                    if v.sum() > 0:
                        rt_i = -math.log(random.random())/v.sum()/v0
                        if rt_i < rt_min:
                            rt_min = rt_i
                            v_hop = v
                            drt_hop = drtn
                            hop_ini = np.array([x, y, z])
    #Above loop finds the carrier that hops. 
    #Yet we still need the hopping direction.
    rdm2 = random.random()
    for i in range(len(v_hop)):
        if (rdm2 > v_hop[:i].sum()/v_hop.sum()) and\
            (rdm2 <= v_hop[:i+1].sum()/v_hop.sum()):
                hop_finl = drt_hop[i]
                break    
# the boundary of carrier_3d would be set again later.
    sys_time += rt_min 
    if rt_min == 1000: 
        print("Error!")
    return sys_time, hop_ini, hop_finl                
#-------------------------------------------------------------------#        
#Vd = -10V, Vs = 0, Vg = -10V, other boundary values are 0
#Use finite difference to solve the potential(y,z)
#Since the boundary potential is given, we need to calc a (30-2)x(200-2) matrix
def get_potential_BGBC(vg, vd, len_x, len_y, len_z, t_ox):
    #solve initial potential
    pot_5544 = np.zeros((len_z-2)*(len_y-2))#should be 5544, means 28*198 unknown
    para_mat = np.zeros(((len_z-2)*(len_y-2), (len_z-2)*(len_y-2)))#5544, 5544
    potential_2d = np.zeros((len_z, len_y))#z is the row num, y is the column num. For the convenience of computation below.
    for i in range(0, len_y-2):
        pot_5544[i] = -vg
    for j in range(t_ox-1, len_z-2, 1):
        pot_5544[j*(len_y-2)] = -vd
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
    for i in range(1, len_z-1):
        potential_2d[i, 1:(len_y-1)] = pot_sol[(i-1)*(len_y-2):i*(len_y-2)]
    potential_2d = boundary_pot(potential_2d)
    return potential_2d
#-------------------------------------------------------------------------------#
while True:
    asw = input('Continue with last simulation? Y/N\n')
    asw = asw.lower()
    if asw == 'y':
        time_counter = input('Please enter time_counter:')
        carrier_3d = np.load('{}\\carrier_3d_{}.npy'.format(data_path, time_counter))
        potential_3d = np.load('{}\\potential_3d_{}.npy'.format(data_path, time_counter))
        time_counter = int(time_counter)
        set_time = int(input('Please input set_time:'))
        if set_time <= time_counter:
            print("set_counter must be large than time_counter!")
        else:
            break
    elif asw == 'n':
        set_time = int(input('Please input set_time:'))
        carrier_3d = np.zeros((len_x, len_y, len_z), dtype = int)
        potential_3d = np.zeros((len_x, len_y, len_z))
        #q_num = np.zeros((len_z, len_y), dtype = int)
        # define the source/drain.
        #1 means a carrier occupied the lattice point. 0 means no carrier
        carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1 #Source side 
        potential_2d = get_potential_BGBC(vg, vd, len_x, len_y, len_z, t_ox)
        potential_3d = update_pot(potential_2d, potential_3d)
        show_mat(potential_2d)
        break
    else:
        print("Not recognized input")
visualize(carrier_3d)
#pot_record = []
current_record = []
time_record = []
#start hopping
while time_counter <= set_time:# set the running time of the simulation
    sys_time, hop_ini, hop_finl = hopping(sys_time, carrier_3d, potential_3d, (len_x, len_y, len_z, t_ox))
    hop_finl = hop_finl.astype(int)
    carrier_3d[hop_ini[0], hop_ini[1], hop_ini[2]] = 0
    carrier_3d[hop_finl[0], hop_finl[1], hop_finl[2]] = 1 
    if hop_finl[1]>0:
        #q_num[hop_finl[2], hop_finl[1]] += 1
        potential_3d += single_charge_pot(hop_finl[0], hop_finl[1], hop_finl[2])
    if hop_ini[1]<(len_y-1):
        #q_num[hop_ini[2], hop_ini[1]] -= 1
        potential_3d -= single_charge_pot(hop_ini[0], hop_ini[1], hop_ini[2])
    """
    potential_2d = boundary_pot(potential_2d)
    potential_3d = update_pot(potential_2d, potential_3d)
    """
    if hop_finl[1] == 0:
        current_counter += 1
    carrier_3d[:, len_y-1, t_ox-1:len_z-1] = 1 #Set the boundary again
    carrier_3d[:, 0, t_ox-1:len_z-1] = 0  
    #current = current_counter*q/sys_time 
    if time_counter % 100 == 0:
        print(time_counter)
    #---------record the current at the moment-------#
    if time_counter % 10000 == 0:
        crt_0 = current_counter
        time_0 = sys_time
    if time_counter % 10000 == 1000:
        crt_1 = current_counter
        time_1 = sys_time
        current_record.append((crt_1 - crt_0)/(time_1 - time_0))
        time_record.append(time_counter - 200)       
    #----------------------------------#
    if time_counter == set_time//10 \
    or time_counter == set_time//5 \
    or time_counter == set_time//2 \
    or time_counter == set_time - 1:
#        pot_record.append(potential_2d)
        show_mat(potential_2d)
        #plt.savefig('potential'+ str(time_counter) + '.png')
        visualize(carrier_3d)
        savenpy(carrier_3d, "carrier_3d", time_counter)
        savenpy(potential_3d, "potential_3d", time_counter)
#        savenpy(potential_2d, "potential_2d", time_counter)
        savenpy(current_record, "current_record", time_counter)
        savenpy(time_record, "time_record", time_counter)
        if len(current_record) > 3:
            diff_1 = abs(current_record[-1] - current_record[-2])/current_record[-2]
            diff_2 = abs(current_record[-2] - current_record[-3])/current_record[-2]
            if diff_1 < 0.02 and diff_2 <0.02:
                break
    time_counter += 1
#--------------------------------------------------------------------#
end = time.time()
print("Runtime: %0.4f"%(end - begin))      
            
        


