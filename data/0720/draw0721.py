from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

pre_traj = '/home/roboticslab/yxj/frankapy/data/0720/my_haptic_subscriber_hololens/'
# pre_traj = 'C:/Users/yan/Documents/GitHub/frankapy/data/0720/my_haptic_subscriber_hololens/'
traj_path = pre_traj+'traj.pkl'  

class MyConstants(object):
    """
        @ Class: MyConstants
        @ Function: get all the constants in this file
    """
    FX_HAT = 2337.218017578125
    FY_HAT = 2341.164794921875
    U0 = 746.3118044533257
    V0 = 564.2590475570069
    # CARTESIAN_CENTER = np.array([0.2068108842682527, 0.611158320250102, 0.1342875493162069])
    # CARTESIAN_CENTER = np.array([-0.0068108842682527, 0.611158320250102, 0.1342875493162069])
    CARTESIAN_CENTER = np.array([0.17, 0.63, 0.19])
    IMG_W = 1440
    IMG_H = 1080

desired_position_bias = np.array([-200, -100])

def plot_transparent_cube(ax,alpha_ = 0.1,x=10,y=20,z=30,dx=40,dy=50,dz=60):
    xx = np.linspace(x,x+dx,2)
    yy = np.linspace(y,y+dy,2)
    zz = np.linspace(z,z+dz,2)

    xx2,yy2 = np.meshgrid(xx,yy)
    ax.plot_surface(xx2,yy2,np.full_like(xx2,z),alpha=alpha_,color='r')
    ax.plot_surface(xx2,yy2,np.full_like(xx2,z+dz),alpha=alpha_,color='r')

    yy2,zz2 = np.meshgrid(yy,zz)
    ax.plot_surface(np.full_like(yy2,x),yy2,zz2,alpha=alpha_,color='r')
    ax.plot_surface(np.full_like(yy2,x+dx),yy2,zz2,alpha=alpha_,color='r')

    xx2,zz2 = np.meshgrid(xx,zz)
    ax.plot_surface(xx2,np.full_like(yy2,y),zz2,alpha=alpha_,color='r')
    ax.plot_surface(xx2,np.full_like(yy2,y+dy),zz2,alpha=alpha_,color='r')



with open(traj_path, 'rb') as f1:
    data2 = pickle.load(f1)
    print(data2[0]['skill_state_dict'].keys())
    q = data2[0]['skill_state_dict']['q']
    O_T_EE = data2[0]['skill_state_dict']['O_T_EE']
    time_list = data2[0]['skill_state_dict']['time_since_skill_started']

    # plot q
    plt.figure()
    plt.plot(time_list,q)
    plt.title('joint angle')
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.savefig(pre_traj+'q.jpg')

    # plot position
    print(O_T_EE.shape)
    position_list, quat_list = [], []
    for i in range(O_T_EE.shape[0]):
        position = O_T_EE[i,[3,7,11]].tolist()
        position_list.append(position)
        rotation_matrix = np.zeros((3,3))
        rotation_matrix[0,:] = O_T_EE[i,0:3]
        rotation_matrix[1,:] = O_T_EE[i,4:7]
        rotation_matrix[2,:] = O_T_EE[i,8:11]
        # print(rotation_matrix)
        ori = R.from_matrix(rotation_matrix)
        quat = ori.as_quat()[[3, 0, 1, 2]]
        quat_list.append(quat)

    plt.figure()
    ax1 = plt.axes(projection='3d')
    position_array  = np.array(position_list)
    ax1.plot3D(position_array[:,0],position_array[:,1],position_array[:,2],label='traj')
    ax1.scatter(position_array[0,0],position_array[0,1],position_array[0,2],c='r',label='initial')
    # ax1.scatter(position_array[200,0],position_array[200,1],position_array[200,2],c='b',label='t=5s')
    # ax1.scatter(MyConstants.CARTESIAN_CENTER[0],MyConstants.CARTESIAN_CENTER[1],MyConstants.CARTESIAN_CENTER[2],c='g',label='goal region center')
    # print(position_list)
    ax1.legend(['traj','initial'])
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('demonstration trajectory 3D')
    plt.savefig(pre_traj+'demonstration_3d.jpg')



    # plot quat
    plt.figure()
    plt.plot(time_list, quat_list)
    plt.title('quaternion vs time')
    plt.savefig(pre_traj+'demonstration_quat.jpg')

    plt.show()
