# -*- coding: utf-8 -*-
"""

This function is used to convert the tip position to the configuration variables 
using inverse kinematics

Input:
   x, y, z   - Tip position  

Output:
   s         - section length
   kappa     - curvature
   phi       - angle of curvature
   
Ref:
@inproceedings{neppalli2008geometrical,
  title={A geometrical approach to inverse kinematics for continuum manipulators},
  author={Neppalli, Srinivas and Csencsits, Matthew A and Jones, Bryan A and Walker, Ian},
  booktitle={2008 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={3565--3570},
  year={2008},
  organization={IEEE}
}
"""
import math

def skp(x, y, z):
    
    s = math.acos(1-2*((x*x+y*y+0.0001)/(x*x+y*y+z*z)))*((x*x+y*y+z*z)/(2*math.sqrt(x*x+y*y+0.0001)))
    kappa = math.acos(1-2*((x*x+y*y+0.0001)/(x*x+y*y+z*z)))/s
    phi = -math.atan2(y, x)

    if kappa ==0:
       kappa = 0.00001
    
    return s, kappa, phi
    