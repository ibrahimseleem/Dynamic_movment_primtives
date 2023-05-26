# -*- coding: utf-8 -*-
"""
 Get the equidistant centers and widths of a set of basis functions
 Input:
   n_basis_functions - number of basis functions
   time              - length of the movement in time
   alpha             - canonical system parameter
 Output:
   centers - centers of the basis functions
   widths  - widths of the basis functions
   
   Cnters and widths are computed based on equation (9)
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

def  centers(basis_functions, tau, alpha_s):
    
    # Basis functions centers are approximately equidistantly spaced in phase space 
    # Centers in time space

    centers_time = tau/(basis_functions-1)*(range(basis_functions+1))
    
    # Compute centers in phases space from equidistant centers in time space
    centers = np.exp(-alpha_s*centers_time/tau)
    
    # compute widths
    widths = np.zeros(len(centers)-1)
    for j in range(len(centers)-1):
        widths[j] = 0.5*np.absolute(centers[j+1]-centers[j])
    
    # Remove the extra center used to compute the widths with diff
    centers = centers[: len(centers)-1]
    return centers, widths