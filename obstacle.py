# -*- coding: utf-8 -*-
"""
 This function is used to avoid obstacle based on coupling term
 Input:
   x          - tip position
   v          - tip velocity
   o          - obstacle position
 Output:
 P            - the repulsive term to keep the robot's tip away from
 obstacle
 
 Equation (6)
 """

import numpy as np
from numpy.linalg import norm
import math
from scipy.spatial.transform import Rotation 

def avoid(x, v, o):
    zeta = 500       # gain 
    var_zeta = 1     # gain
    
    o = o.reshape(-1,1)
    
    for i  in range(o.shape[1]): # o.reshape(-1,1).shape[1] number of obstacles
        u = o[:, i] - x  # distance from obstacle to the tip's position
        
    # Steering angle equation(7)
    # I used here atan2 instead of arccos to determine exactly the + or – angle relative to the way we’re headed.
    phi = math.atan2(np.dot(u, v), np.linalg.norm(np.cross(u, v)))
    a = np.cross(u, v)
    if all(element == 0 for element in a):
       r = np.diag(np.ones(3))
    else:
        r = Rotation.from_rotvec(math.pi * a).as_matrix()
    
    # compute the repulsive term
    P = zeta* r@ v.reshape(-1, 1)* phi* np.exp(-var_zeta*phi)
    
    return P
       