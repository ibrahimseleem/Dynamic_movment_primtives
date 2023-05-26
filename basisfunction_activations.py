# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:54:27 2023

@author: seleem
"""

# -*- coding: utf-8 -*-
"""
 Compute basis activations for 1 or more time steps
 Input:
   basis_function -  number of basis functions
   centers - centers of the basis functions
   widths  - widths of the basis functions
   s      - if scalar: current phase (or time)
             if vector: sequence of phases (or time)
 Output:
  activations - activations of the basis functions at each time step
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

def  activations(basis_function, centers, widths, s):
    
     activ = np.zeros(basis_function)
     
     for k in range(basis_function):
         activ[k] = np.exp((-0.5/(widths[k]**2))*(s - centers[k])**2) # equation (9)
    
     return activ

    