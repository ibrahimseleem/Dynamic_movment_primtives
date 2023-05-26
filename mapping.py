# -*- coding: utf-8 -*-
"""
Mapping from configuration parameters to task space

 INPUT:
   s         - section length
   kappa     - curvature
   phi       - angle of curvature
   sect_points: points per section

OUTPUT:
 T: Transformation matrices

"""

import numpy as np
import math

def trans(s, kappa, phi):
    
    sect_points=50; #number of points per section
    si = np.linspace(0, s, sect_points)
    T = np.zeros((len(si), 16))  # Define T as a 3-dimensional array of shape (len(si), 4, 4)

    c_p = math.cos(phi)
    s_p = math.sin(phi)
    
    for i in range(len(si)):
        s = si[i]
        c_ks = math.cos(kappa * s)
        s_ks = math.sin(kappa * s)
        if kappa == 0:
           T[i, :] = np.array([[c_p * c_ks, s_p * c_ks, -s_ks, 0],
                            [-s_p, c_p, 0, 0],
                            [c_p * s_ks, s_p * s_ks, c_ks, 0],
                            [0, 0, s, 1]]).reshape(1, -1)
        else:
           T[i, :] = np.array([[c_p * c_ks, s_p * c_ks, -s_ks, 0],
                            [-s_p, c_p, 0, 0],
                            [c_p * s_ks, s_p * s_ks, c_ks, 0],
                            [(c_p * (1 - c_ks)) / kappa, -(s_p * (1 - c_ks)) / kappa, s_ks / kappa, 1]]).reshape(1, -1)
    
    return T