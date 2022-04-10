from tkinter.tix import Tree
import numpy as np
import math
import rospy
import argparse
import pickle
# from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

from my_dmp.processing import *
from my_dmp.dmp_class import *

from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_dmp_weights_file_path', '-f', type=str, default='joint_dmp_weights_0408.pkl')
    args = parser.parse_args()

#====================================training

    state_dict = pickle.load( open( '/home/roboticslab/yxj/frankapy/traj0408.pkl', "rb" ) )

    cartesian_trajectory = {}
    skill_state_dict = {}

    for key in state_dict.keys():
        skill_dict = state_dict[key]
        
        if skill_dict["skill_description"] == "GuideMode":
            skill_state_dict = skill_dict["skill_state_dict"]
            q = skill_state_dict["q"]
            cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
            time_since_skill_started = skill_state_dict["time_since_skill_started"]
    # print(skill_dict)

    trajectory_start_time = 0.0
    trajectory_end_time = time_since_skill_started[-1]

    x_range = [trajectory_start_time, trajectory_end_time]
    trajectory_times = np.reshape(trajectory_start_time+time_since_skill_started,(-1,1))

    tau = float(0.5/trajectory_end_time)
    alpha = 20
    beta = alpha/4
    num_basis = 4
    num_sensors = 7


    my_dmp = DMPTrajectory(tau,alpha,beta,7,num_basis,num_sensors)
    joint_dmp_weights, _ = my_dmp.train_using_individual_trajectory('joint', trajectory_times, q)
    my_dmp.save_weights("joint_dmp_weights_0408.pkl",joint_dmp_weights)

#====================================run

    print('Starting robot')
    fa = FrankaArm()

    with open(args.joint_dmp_weights_file_path, 'rb') as pkl_f:
        joint_dmp_info = pickle.load(pkl_f)
    if 'initial_sensor_values' in joint_dmp_info:
        print('----------')
        print('initial_sensor_values: ', joint_dmp_info['initial_sensor_values'])
        print('----------')
    print(joint_dmp_info)
    joint_dmp_info['tau'] *= 1.0

    initial_sensor_values = np.linspace(1.0, 1.5, 6)
    # initial_sensor_values = [1]
    # base_initial_sensor_value = np.array([1, 1.5, 1.5, 1.5, 1, 1.5, 1])
    base_initial_sensor_value = np.ones(7,)
    initial_joint_error = np.zeros((len(initial_sensor_values), 7))
    for iter, current_initial_sensor_value in enumerate(initial_sensor_values):
        print('==============================>')
        print('round: (%d/%d) | init sensor val: %.2f' % (iter, len(initial_sensor_values), current_initial_sensor_value))
        print('Resetting robot to home joints!')

        fa.reset_pose(block=True)
        current_home_joints = fa.get_joints()

        print('Home joints: ', current_home_joints)

        print('Resetting robot to home joints!')

        sensor_values = [1] + [current_initial_sensor_value for i in range(3)] + [1, 1, 1]
        sensor_values = np.array(sensor_values)

        fa.execute_joint_dmp(joint_dmp_info=joint_dmp_info, 
                             duration=(10/1.0), 
                             initial_sensor_values=(sensor_values).tolist(),
                             block=True)

        initial_joint_error[iter, :] = fa.get_joints() - current_home_joints
        print('The robot has reached the goal!')
        print('(g - y_0) in current iteration: ', initial_joint_error[iter, :])
    np.save('dmp_joint_error_data.npy', initial_joint_error)
    print('The robot stopped!')
    # reproduced_file = 'reproduced_traj0407.pkl'
    # with open(reproduced_file, 'wb') as pkl_f:
    #     pickle.dump(goal, pkl_f)
    #     print("Did save skill dict: {}".format(reproduced_file))