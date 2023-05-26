# -*- coding: utf-8 -*-
"""
 If you use this code in the context of a publication, I would appreciate 
 it if you could cite it as follows:

 @article{seleem2023imitation,
   title={Imitation-Based Motion Planning and Control of a Multi-Section Continuum Robot Interacting With the Environment},
   author={Seleem, Ibrahim A and El-Hussieny, Haitham and Ishii, Hiroyuki},
   journal={IEEE Robotics and Automation Letters},
   volume={8},
   number={3},
   pages={1351--1358},
   year={2023},
   publisher={IEEE}
 }
 }

Please follow the following steps before cimpilng the code:
    
1) pip install --upgrade pip
2) pip install pyquaternion
3) pip install spatialmath-python
4) pip install SciencePlots


"""


from scipy.io import loadmat
import numpy as np
from demonstrations import refer_trajec
from matplotlib import pyplot as plt  # Plot package
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter


# add directory of your data
# the data is organised as folows [time orientation_quaternion position]
# Modify the directory 
data=loadmat(r"E:\ELZERO_PYTHON\matlab\python_codes\TipPose_manipulator.mat") 
tip_pose = data["t_ori_pos"]

time, dt, tip_pos, velocity, acceleration, tip_orien, g0_ori, tau, omega, eta, eta_dot, R0, Rg, Kd = refer_trajec(tip_pose)

basis_functions = 100 # number of basis functions (modify it to improve the reults)


from IbMP_approach import DMP

time,tip_pos, tip_orien, q, x, s, obstacle_position = DMP(time, dt, tip_pos, 
                          velocity, acceleration, 
                          tip_orien, g0_ori, tau, 
                          omega, eta, eta_dot, R0, 
                          Rg, Kd, basis_functions)



plt.figure()
ax = plt.axes(projection="3d")
ax.plot(tip_pos[:,0], tip_pos[:,1], tip_pos[:,2], color='blue')
ax.plot(x[:,0], x[:,1], x[:,2], color='red')
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.set_zlabel("Z [cm]")
plt.show(block=True)

# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2)

# Plot the data in each subplot
axs[0, 0].plot(time, tip_orien[:, 0], color='blue')
axs[0, 0].plot(time, q[:, 0], color='red')

axs[0, 1].plot(time, tip_orien[:, 1], color='blue')
axs[0, 1].plot(time, q[:, 1], color='red')

axs[1, 0].plot(time, tip_orien[:, 2], color='blue')
axs[1, 0].plot(time, q[:, 2], color='red')

axs[1, 1].plot(time, tip_orien[:, 3], color='blue')
axs[1, 1].plot(time, q[:, 3], color='red')
# Adjust the spacing between subplots
plt.tight_layout()

# Show the subplot
plt.show()


## Animation
#actual tip_position at 0 position
x_act = x[:,0] - tip_pos[0,0]
y_act = x[:,1] - tip_pos[0,1]
z_act = x[:,2]

# Create the figure and set title, labels, and limits
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Set the view
ax.view_init(elev=50, azim=10)


lmax = 16.5  
ax.set_xlim([-lmax, lmax])
ax.set_ylim([-lmax, lmax])
ax.set_zlim([0, 2*lmax])
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')
ax.set_title('Time: {:.2f} sec'.format(time[0]), fontsize=14)



p, = ax.plot3D([x_act[0]], [y_act[0]], [z_act[0]], color='red', linewidth=2)
m = ax.scatter([x_act[0]], [y_act[0]], [z_act[0]], color='black', marker='o')  # start position
z = ax.scatter([x_act[len(tip_pos)-1]], [y_act[len(tip_pos)-1]], [z_act[len(tip_pos)-1]], color='c', marker='x')  # start position


T1_ccn = np.zeros((50, 3))
radius1 = 4
k, = ax.plot3D(T1_ccn[:, 0], T1_ccn[:, 1], T1_ccn[:, 2], linewidth=radius1, color='b')


def drawframe(i):
    from configuration_vars import skp  # get configuration variables
    s, kappa, phi = skp(x_act[i], y_act[i], z_act[i])
    
    from mapping import trans  # Mapping from configuration parameters to task space
    T1_cc = trans(s, kappa, phi)  # Tip position
    
    # Update the trajectory plot
    p.set_data_3d(x_act[0:i], y_act[0:i], z_act[0:i])

    m.set_offsets(np.array([[x_act[i], y_act[i], z_act[i]]]))

    # Update the point
    k.set_data_3d(T1_cc[:, 12], T1_cc[:, 13], T1_cc[:, 14])
    
    # print sampling time
    ax.set_title(r'Time: {:.2f} sec'.format(time[i]), fontsize=12)

    return (p, m, k)


# Animation part
anim = animation.FuncAnimation(fig, drawframe, frames=len(tip_pos), interval=20, blit=True )
# Save the animation as GIF
anim.save("anim.gif", writer=PillowWriter(fps=20))