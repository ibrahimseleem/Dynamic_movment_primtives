# -*- coding: utf-8 -*-
"""
Computation of the learning weights

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

def weights_computation(tip_pos, velocity, acceleration, omega, eta, eta_dot, s, tau, R0, Rg, Kd, basis_function, alpha_z, alpha_z1, beta_z, beta_z1, alpha_s, x0, xg):
    
    # Comoute centers and widths
    from basisfunction_centers_widths import centers
    centers, widths = centers(basis_function, tau, alpha_s)
    
    # compute activations
    from basisfunction_activations import activations
    activ = activations(basis_function, centers, widths, s)
    
    # compute the target forces, please check equation (4) and (10)
    f  = (-alpha_z*(beta_z*(xg-tip_pos)-velocity) + tau*acceleration)/(xg-x0)
    fq = np.transpose(np.dot(np.linalg.inv(Kd),(tau*eta_dot+alpha_z1*eta-alpha_z1*beta_z1*omega).reshape(-1, 1)))

    # Compute the regression, using linear least squares
    # (http://en.wikipedia.org/wiki/Linear_least_squares)
    vs_repmat = np.tile(s, basis_function).reshape(-1, 1)
    sum_activations = np.tile(sum(np.absolute(activ)), basis_function).reshape(-1, 1)
    activations_normalised = activ.reshape(-1, 1) /sum_activations
    vs_activ_norm = vs_repmat * activations_normalised
    small_diag = 0.01 * np.diag(np.ones(basis_function), 0)  # To overcome singularity
    AA = np.linalg.inv(vs_activ_norm*vs_activ_norm.reshape(1, -1) + small_diag) @ vs_activ_norm
    # compute the learned weight
    W = AA* f  # Equation (9)
    psi = AA* fq  # Equation (12)
    return activ, W, psi
