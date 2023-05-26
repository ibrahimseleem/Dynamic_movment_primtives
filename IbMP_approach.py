# -*- coding: utf-8 -*-
"""
Dynamic movement primitives function

Input:
   Reference position and orientation  

Output:
   Actual tip's pose


"""

import numpy as np
from numpy.linalg import norm
import math
from scipy.io import loadmat
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from scipy.linalg import logm, expm
from quat2rot import quaternion_rotation_matrix
from spatialmath.base import *

def DMP(time, dt, tip_pos, velocity, acceleration, tip_orien, g0_ori, tau, omega, eta, eta_dot, R0, Rg, Kd, basis_functions):
    
    ### tip_position parameters
    x0 = tip_pos[0]                 # initial position
    xg = tip_pos[len(tip_pos)-1]    # goal_position
    # gn = xg0                         # continious goal
    
    x     =   np.vstack((tip_pos[0], np.zeros((len(tip_pos)-1, tip_pos.shape[1]))))        # create array of actual tip position
    v     =  np.vstack((velocity[0], np.zeros((len(tip_pos)-1, tip_pos.shape[1]))))        # array of actual velocity 
    v_dot =  np.vstack((acceleration[0], np.zeros((len(tip_pos)-1, tip_pos.shape[1]))))    # array of actual accelerarion

    #learning gains
    alpha_z = 200
    beta_z  = alpha_z/4
    

    ### tip_orientation parameters
    q       = np.vstack((tip_orien[0, :], np.zeros((len(tip_orien)-1, tip_orien.shape[1]))))               # initial of actual orientation
    omega_a = np.vstack((omega[0, :], np.zeros((len(tip_pos)-1, tip_pos.shape[1]))))
    eta_a   = np.vstack((eta[0, :], np.zeros((len(tip_pos)-1, tip_pos.shape[1]))))   

    #learning gains
    alpha_z1 = 500
    beta_z1  = 450 
    
    s = np.zeros(len(tip_pos)) 

    ## start learning algorithm
    for i in range(len(tip_pos)-1):
    # for i in range(1):
        # computation of phase stopping variable
        ex0 = norm(xg - x[i])  
        ex  = norm(tip_pos[i] - x[i]) 
        
        # print(Quaternion([1,0,0,0]))
        
        if (Quaternion(g0_ori)*Quaternion(q[i,:]).conjugate).normalised == Quaternion(1,0,0,0):
            eq0 = 2 * math.pi
        else :
            eq0 = norm(2*Quaternion.log((Quaternion(g0_ori)*Quaternion(q[i,:]).conjugate).normalised))

        if (Quaternion(tip_orien[i,:])*Quaternion(q[i,:]).conjugate).normalised == Quaternion(1,0,0,0):
            eq = 2 * math.pi
        else :
            eq = norm(2*Quaternion.log((Quaternion(tip_orien[0])*Quaternion(q[i,:]).conjugate).normalised))
        
        exq = ex0 + ex + eq0 + eq  # error signal to synchronise poisition and orientation (equation. 13)
     
        from canonicalsystem import canonical    # canonical function
        alpha_s = 1.4
        alpha_r = 0.001

        s[i] = canonical(time[i], tau, exq, alpha_s, alpha_r)
        
        # compute target forces and learned weights
        from transformation_train import weights_computation
        activ, W, psi = weights_computation(tip_pos[i,:], velocity[i,:], acceleration[i,:], omega[i,:], 
                                          eta[i,:], eta_dot[i,:], s[i], tau, R0, Rg, Kd, basis_functions, 
                                          alpha_z, alpha_z1, beta_z, beta_z1, alpha_s, x0, xg)
        
 
        # compute the actual forces
        # Tip's position
        weighted_sum_activations_pos = activ @ W
        sum_activations = sum(activ)
        f = (weighted_sum_activations_pos/sum_activations)*s[i]
        
        
        # obstacle avoidance part Equation (4)
        obstacle = True    
        # obstacle position (can be changed)
        obs = np.array([11.2569041108792,	9.86753598033462,	28.1175589891711])
            
        
        from obstacle import avoid
        if obstacle == False:
           P = avoid(x[i], v[i], obs)
           v_dot[i+1,:] = (alpha_z*(beta_z*(xg - x[i,:]) - v[i, :]) + (f * (xg-x0)) + P.reshape(1, -1))/tau #Ka = alpha_z*beta_z, Da = beta_z
        else:
            v_dot[i+1,:] = (alpha_z*(beta_z*(xg - x[i,:]) - v[i, :]) + (f * (xg-x0)))/tau
        
        # Continious goal Equation(14)
        alpha_g = 10   # gain
        k       = 0.01      # gain Equation (15)
        xg_dot  = alpha_g*(xg-xg)/tau  #Equation (14)
        xg      = xg + dt*xg_dot            # Integrate Equation (14)
        
        # Integrate acceleration to get velocity and position
        v[i+1,:] = v[i, :] + dt*v_dot[i+1,:]
        x[i+1,:] = x[i,:]  + dt*v[i,:]
        
        # Tip's orientation
        weighted_sum_activations_ori = np.dot(activ, psi)
        fq = np.dot(Kd, (weighted_sum_activations_ori.reshape(-1, 1)/sum_activations)*s[i] ).reshape(1, -1)      
       
        eta_adot = (alpha_z1*(beta_z1*omega_a[i,:] - eta_a[i,:]) + fq)/tau #Equation(10) K_z = alpha_z1*beta_z1, D_z = beta_z1
        eta_a[i+1,:] = eta_a[i,:] + dt*eta_adot
        
        # Compute actual orientation Equation (11)
        xx = (2*0.01*Quaternion.log((Quaternion(tip_orien[0,:])*Quaternion(q[i,:]).conjugate).normalised) + 
              Quaternion.log((Quaternion(tip_orien[i,:])*Quaternion(q[i,:]).conjugate).normalised))
        r = dt*(np.insert(eta_a[i+1,:], 0, 0) + np.array([xx.w, xx.x, xx.y, xx.z]))/(2*tau)

                
        # add step disturbance to the orientation trajectory
        add_dis = False
        yy = Quaternion.exp(Quaternion(r))*Quaternion(q[i,:]).normalised

        if add_dis == True:
           if 2 <= time[i] and time[i] >= 5:  # period od disturbance
              dis = 0.01
              q[i+1, :] =  np.array([yy.w, yy.x, yy.y, yy.z]) + dis    
        else:
            dis = 0
            q[i+1, :] =  np.array([yy.w, yy.x, yy.y, yy.z]) + dis
            
       
        # compute actual values of omega and eta
        zz = Quaternion.log((Quaternion(g0_ori)*Quaternion(q[i+1, :]).conjugate).normalised)
        omega_a[i+1, :] = 2*np.array([zz.x, zz.y, zz.z])
        eta_a[i+1, :]   = tau* omega_a[i+1, :] 

    return time,tip_pos, tip_orien, q, x, s, obs
