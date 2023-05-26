# -*- coding: utf-8 -*-
"""
 Integrate a canonical system 
 Input:     
   alpha_s    - gain
   alpha_r    - gain
   exq      - tracking error
 Output:
   s - phase stopping variable
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


def canonical(t, tau, exq, alpha_s, alpha_r):
    s = np.exp(-alpha_s*t/(tau*(1+(alpha_r*exq))))

    return s
    
