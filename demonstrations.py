"""
Input:
   tip_pose      - [samle time, orientation[quaternion], position]  

Output:
   velocity      - estimated velocity
   acceleration  - estimated acceleration 
   eta, eta_dot  - he quaternion derivates 
   
   """



import numpy as np
from numpy.linalg import norm
import math
from scipy.io import loadmat
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm
from quat2rot import quaternion_rotation_matrix
from spatialmath.base import *


def refer_trajec(tip_pose):
    #extract data from .mat file
    time = tip_pose[:,0]            # Run time
    dt = time[1]-time[0]            # sampling time
    tip_orien = tip_pose[:,1:5]     # tip position     
    tip_pos = tip_pose[:,5:]        # tip_otientation 
    #print(tip_pos.shape) 
    #print(type(tip_pos)) 

    # compute velocity and acceleration
    velocity =  np.zeros((len(tip_pos)-1, tip_pos.shape[1])) # create empty array
    acceleration =  np.zeros((len(tip_pos)-1, tip_pos.shape[1])) # create empty array
    
    vel_init = np.array([0,0,0])      # initial velocity
    acc_init = np.array([0,0,0])      # initial acceleration
    

    # Compute velocity

    for countv in range(len(tip_pos)-1):
        velocity [countv, :] = (tip_pos[countv+1, :] - tip_pos[countv, :])/time[1]
        
    velocity = np.vstack((vel_init, velocity))
    
    # Compute acceleration
    for counta in range(len(velocity)-1):
        acceleration [counta, :] = (velocity[counta+1, :] - velocity[counta, :])/time[1]
       
    acceleration = np.vstack((acc_init, acceleration))
    
    # orientation part
    g0_ori = tip_orien[len(tip_orien)-1, :]    # goal orientation
    tau = time[len(tip_orien)-1]              # temporal scaling factor
    
    # Calaculation of eta and eta_dot 
    
    y =  np.zeros((len(tip_orien), tip_orien.shape[1])) # create empty array
    omega =  np.zeros((len(tip_orien), tip_pos.shape[1])) # create empty array
    eta =  np.zeros((len(tip_orien), tip_pos.shape[1])) # create empty array

    for countx in range(len(tip_orien)):
        x = 2*Quaternion.log((Quaternion(g0_ori)*Quaternion(tip_orien[countx]).conjugate).normalised)
        y[countx, :]  = np.array([x.real, x.imaginary[0], x.imaginary[1], x.imaginary[2]])
        omega [countx, :] = y[countx, 1:]    # quaternion with zero scalar
        eta [countx, :] = tau* omega [countx, :]  
        
    
    # Compute eta_dot
    eta_dot =  np.zeros((len(eta)-1, eta.shape[1])) # create empty array
    eta_dot_init = np.array([0,0,0])      # initial acceleration

    for count in range(len(eta)-1):
        eta_dot [count, :] = (eta[count+1, :] - eta[count, :])/time[1]

    eta_dot = np.vstack((eta_dot_init, eta_dot))
    
    R0 = quaternion_rotation_matrix(tip_orien[0])
    Rg = quaternion_rotation_matrix(tip_orien[len(tip_orien)-1])

    Kd=np.diag(vex(logm(Rg @ np.transpose(R0))))
    
 
    return time, dt, tip_pos, velocity, acceleration, tip_orien, g0_ori, tau, omega, eta, eta_dot, R0, Rg, Kd

