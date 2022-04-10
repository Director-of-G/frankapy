from tkinter.tix import Tree
import numpy as np
import time
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
    joint_dmp_info['tau'] *= 1.5

    initial_sensor_values = 1.0
    # initial_sensor_values = [1]

    fa.execute_joint_dmp(joint_dmp_info=joint_dmp_info, 
                         duration=(10/1.5), 
                         initial_sensor_values=(1.0 * np.ones(7,)).tolist(),
                         block=False)

    joints_memory = []

    start_time = time.time()

    while True:
        joints = fa.get_joints()
        print(joints)
        joints_memory.append(joints.tolist())
        if time.time() - start_time > 10:
            break
        time.sleep(0.05)

    plt.figure()
    plt.plot(joints_memory)
    plt.show()
    
    # reproduced_file = 'reproduced_traj0407.pkl'
    # with open(reproduced_file, 'wb') as pkl_f:
    #     pickle.dump(goal, pkl_f)
    #     print("Did save skill dict: {}".format(reproduced_file))