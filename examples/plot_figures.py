import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import copy
import pickle
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from my_dmp.dmp_class import *

from my_dmp.processing import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

SAVE_PATH_PRE_FIX = './data/0415'

if __name__ == '__main__':  

    parser = argparse.ArgumentParser()

    # path constants
    parser.add_argument('--traj_path', '-tp', type=str, default=SAVE_PATH_PRE_FIX + '/traj_0415.pkl',
                        help='Path of the expert trajectory.')
    parser.add_argument('--memory_path', '-mp', type=str, default=SAVE_PATH_PRE_FIX + '/joints_memory.txt',
                        help='Path of real Franka joint angles, recorded by Franka Interface.')
    parser.add_argument('--dmp_weights_path', '-wp', type=str, default=SAVE_PATH_PRE_FIX + '/joint_dmp_weights.pkl',
                        help='Path of the calculated dmp weight to save.')
    parser.add_argument('--save_path', '-sp', type=str, default=SAVE_PATH_PRE_FIX)

    # frequency constants
    parser.add_argument('--record_frequency', '-rf', type=float, default=10,
                        help='Frequency that the we recorded the expert trajectory, \
                              see my_haptic_subscriber.py for details.')
    parser.add_argument('--execute_frequency', '-ef', type=float, default=1000,
                        help='Frequency that franka interface compute joint angles y, \
                              see franka_interface: joint_dmp_trajectory_generator.cpp for details.')

    # other constants
    parser.add_argument('--run_time', '-rt', type=float, default=21.0)
    parser.add_argument('--tau', '-tau', type=float, default=1.0)

    args = parser.parse_args()

    #====================================plot expert trajectory
    traj_path = '/home/roboticslab/yxj/frankapy/data/0415/traj_0415.pkl'  
    state_dict = pickle.load(open(traj_path, "rb" ) )

    for key in state_dict.keys():
        skill_dict = state_dict[key]
        
        if skill_dict["skill_description"] == "GuideMode":
            skill_state_dict = skill_dict["skill_state_dict"]
            my_goal = skill_state_dict["q"][-1, :]
            my_y0 = skill_state_dict["q"][0, :]
            q = skill_state_dict["q"]
            cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
            time_since_skill_started = skill_state_dict["time_since_skill_started"]
    print(cartesian_trajectory)
    # trajectory_start_time = 0.0
    trajectory_end_time = time_since_skill_started[-1]


    plt.figure()
    plt.plot(time_since_skill_started, q)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.grid()
    plt.title('expert trajectory vs time')

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(cartesian_trajectory[:,0],cartesian_trajectory[:,1],cartesian_trajectory[:,2],label='traj')
    ax1.scatter(cartesian_trajectory[0,0],cartesian_trajectory[0,1],cartesian_trajectory[0,2],'x',label='initial')
    ax1.scatter(cartesian_trajectory[-1,0],cartesian_trajectory[-1,1],cartesian_trajectory[-1,2],'o',label='end')
    ax1.legend()
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('expert trajectory 3D')

    #======================= plot reproduced trajectory
    # train joint dmp
    dt = 1 / len(q)  # set dt for dmp and cs = (1 / traj_data_points)
    dmp_trajectory = MyDMPTrajectory(dt=dt, num_dims=7, num_basis=42)
    weights = dmp_trajectory.train(q.transpose(1, 0))

    joint_dmp_info = {'mu': dmp_trajectory.mu_all,
                      'h': dmp_trajectory.h_all,
                      'weights': weights,
                     }

    # save joint dmp weights
    # save_path = args.dmp_weights_path
    # with open(save_path, 'wb') as pkl_f:
    #     pickle.dump(joint_dmp_info, pkl_f, protocol=2)
    
    
    # reproduce trajectory
    tau_reproduce = args.tau
    freqency_scale = args.record_frequency / args.execute_frequency
    time_steps_reproduce = round(dmp_trajectory.cs.time_steps / (tau_reproduce * freqency_scale))
    y, dy, ddy = dmp_trajectory.execute(tau=(tau_reproduce * freqency_scale))
    reproduce_trajectory_time = np.linspace(0,
                                            time_since_skill_started[-1] / tau_reproduce,
                                            time_steps_reproduce)
    
    plt.figure()
    plt.plot(reproduce_trajectory_time, y)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.grid()
    plt.title('dmp learned trajectory vs time')

    #===========================================plot executed trajectory
    joints_memory = np.load(args.save_path + '/joints_angle_memory_0415.npy')
    execution_time = np.load(args.save_path + '/execution_time_0415.npy')
    pose_trans_memory = np.load(args.save_path + '/ee_trans_memory_0415.npy')
    pose_rot_memory = np.load(args.save_path + '/ee_rot_memory_0415.npy')

    plt.figure()
    plt.plot(execution_time[execution_time<=30], joints_memory[execution_time<=30])
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.grid()
    plt.title('dmp executed joint trajectory vs time')

    plt.figure()
    plt.plot(execution_time[execution_time<=30], pose_trans_memory[execution_time<=30])
    plt.legend(['x','y','z'])
    plt.xlabel('time/s')
    plt.ylabel('position/m')
    plt.grid()
    plt.title('dmp executed ee trajectory vs time')

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(pose_trans_memory[:,0],pose_trans_memory[:,1],pose_trans_memory[:,2],label='traj')
    ax1.scatter(pose_trans_memory[0,0],pose_trans_memory[0,1],pose_trans_memory[0,2],c='r',label='initial')
    ax1.scatter(pose_trans_memory[-1,0],pose_trans_memory[-1,1],pose_trans_memory[-1,2],c='g',label='end')
    ax1.legend()
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('executed trajectory 3D')

    #===========================================plot executed trajectory 2
    args.save_path = SAVE_PATH_PRE_FIX+'/tau_1-2'
    joints_memory = np.load(args.save_path + '/joints_angle_memory_0415.npy')
    execution_time = np.load(args.save_path + '/execution_time_0415.npy')
    pose_trans_memory = np.load(args.save_path + '/ee_trans_memory_0415.npy')
    pose_rot_memory = np.load(args.save_path + '/ee_rot_memory_0415.npy')

    plt.figure()
    plt.plot(execution_time[execution_time<=20], joints_memory[execution_time<=20])
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.grid()
    plt.title('dmp executed joint trajectory vs time')

    plt.figure()
    plt.plot(execution_time[execution_time<=20], pose_trans_memory[execution_time<=20])
    plt.xlabel('time/s')
    plt.ylabel('position/m')
    plt.grid()
    plt.title('dmp executed ee trajectory vs time')

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(pose_trans_memory[:,0],pose_trans_memory[:,1],pose_trans_memory[:,2],label='traj')
    ax1.scatter(pose_trans_memory[0,0],pose_trans_memory[0,1],pose_trans_memory[0,2],c='r',label='initial')
    ax1.scatter(pose_trans_memory[-1,0],pose_trans_memory[-1,1],pose_trans_memory[-1,2],c='g',label='end')
    ax1.legend()
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('executed trajectory 3D')

    plt.show()
