from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb

pre_traj = '/home/roboticslab/yxj/frankapy/data/0723/my_adaptive_control_20220723_171220_with_Js_no_update/'
print(pre_traj)
traj_path = pre_traj+'data.pkl'  

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

    f_list = data2['f_list']
    p_list = data2['p_list']
    kesi_x_list = data2['kesi_x_list']
    pixel_1_list = data2['pixel_1_list']
    pixel_2_list = data2['pixel_2_list']
    time_list = data2['time_list']
    q_and_manipubility_list = data2['q_and_manipubility_list']
    quat_list = data2['quat_list']
    f_quat_list = data2['f_quat_list']
    p_quat_list = data2['p_quat_list']
    kesi_rall_list = data2['kesi_rall_list']
    position_list = data2['position_list']
    Js_list = data2['Js_list']

    plt.figure()
    plt.plot(time_list)

    plt.show()
    time_array = np.array(time_list)
    # print(np.mean(time_array[1:200] - time_array[:199]))
    # print(np.mean(time_array[270:] - time_array[269:-1]))
    exit()

    # vision part
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_list, color='b',label = 'f')
    plt.plot(time_list, p_list, color='r',label = 'P')
    plt.title('f and P for vision region')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list,kesi_x_list,label = 'kesi')
    plt.title('kesi_x')
    # plt.savefig(pre_traj+'f_P_kesi_x.jpg')

    plt.figure()
    plt.plot(time_list, pixel_1_list,color='b',label = 'vision position')
    plt.plot(time_list, pixel_2_list+desired_position_bias,color='r',label = 'desired position')
    plt.legend()
    plt.ylim([0,MyConstants.IMG_W])
    plt.title('vision position vs time')
    # plt.savefig(pre_traj+'vision_position.jpg')

    plt.figure()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b',label = 'vision trajectory')
    plt.scatter((pixel_2_list[0]+desired_position_bias)[0], (pixel_2_list[0]+desired_position_bias)[1],color='r',label = 'desired position')
    plt.xlim([0,MyConstants.IMG_W])
    plt.ylim([0,MyConstants.IMG_H])
    ax = plt.gca()
    ax.invert_yaxis()
    plt.legend()
    plt.title('vision trajectory')
    # plt.savefig(pre_traj+'vision_trajectory.jpg')

    # cartesian part
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_quat_list, color='b',label = 'f_quat')
    plt.plot(time_list, p_quat_list, color='r',label = 'p_quat')
    plt.title('f and p for quaternion')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list, quat_list)
    plt.title('quaternion vs time')
    # plt.savefig(pre_traj+'cartesian_quat.jpg')

    plt.figure()
    ax1 = plt.axes(projection='3d')
    position_array  = np.array(position_list)
    ax1.plot3D(position_array[:,0],position_array[:,1],position_array[:,2],label='traj')
    ax1.scatter(position_array[0,0],position_array[0,1],position_array[0,2],c='r',label='initial')
    # ax1.scatter(position_array[200,0],position_array[200,1],position_array[200,2],c='b',label='t=5s')
    ax1.scatter(MyConstants.CARTESIAN_CENTER[0],MyConstants.CARTESIAN_CENTER[1],MyConstants.CARTESIAN_CENTER[2],c='g',label='goal region center')
    
    c = np.array([0.05, 0.04, 0.05])
    plot_transparent_cube(ax1,0.1,MyConstants.CARTESIAN_CENTER[0]-c[0],MyConstants.CARTESIAN_CENTER[1]-c[1],MyConstants.CARTESIAN_CENTER[2]-c[2],
    2*c[0],2*c[1],2*c[2])
    ax1.legend(['traj','initial','goal region center'])
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('executed trajectory 3D')
    # plt.savefig(pre_traj+'cartesian_3d.jpg')

    plt.figure()
    plt.plot(time_list,np.reshape(kesi_rall_list,(np.shape(time_list)[0],-1)),label = 'kesi')
    plt.legend()
    plt.title('kesi for 6 dimensions')
    # plt.savefig(pre_traj+'cartesian_kesi.jpg')

    plt.figure()
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.plot(time_list,np.array(Js_list)[:,i],label = 'kesi')
    plt.suptitle('Js')
    # plt.savefig(pre_traj+'Js.jpg')

    plt.figure()
    plt.plot(time_list,q_and_manipubility_list[:,0:7])
    plt.legend(['joint 1','joint 2','joint 3','joint 4','joint 5','joint 6','joint 7'])
    plt.title('q')
    plt.savefig(pre_traj+'q.jpg')

    plt.show()