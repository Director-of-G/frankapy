from tkinter.tix import Tree
import numpy as np
import math
import rospy
import argparse
import pickle
# from autolab_core import RigidTransform, Point
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC

from my_dmp.processing import *
from my_dmp.dmp_class import *

from matplotlib import pyplot as plt

def my_make_joint_dmp_info(tau, alpha_y, beta_y, num_dims, num_basis, alpha_x, mu, h, weights):
    return {
        'tau': tau,
        'alpha': alpha_y,
        'beta': beta_y,
        'num_dims': num_dims,
        'num_basis': num_basis,
        'num_sensors': alpha_x,
        'mu': mu if isinstance(mu, list) else mu.reshape(-1).tolist(),
        'h': h if isinstance(h, list) else h.reshape(-1).tolist(),
        'phi_j': np.ones(num_basis),
        'weights': weights if isinstance(weights, list) else weights.tolist(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_dmp_weights_file_path', '-f', type=str, default='joint_dmp_weights_0408.pkl')
    args = parser.parse_args()

    print(FC.DEFAULT_JOINT_IMPEDANCES)


#====================================test_dmp

    state_dict = pickle.load( open('/home/roboticslab/yxj/frankapy/data/0408/traj0408.pkl', "rb" ) )
    my_goal = None
    my_y0 = None
    # cartesian_trajectory = {}
    # skill_state_dict = {}

    for key in state_dict.keys():
        skill_dict = state_dict[key]
        
        if skill_dict["skill_description"] == "GuideMode":
            skill_state_dict = skill_dict["skill_state_dict"]
            my_goal = skill_state_dict["q"][-1, :]
            my_y0 = skill_state_dict["q"][0, :]
            # q = skill_state_dict["q"]
            # cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
            time_since_skill_started = skill_state_dict["time_since_skill_started"]
    
    import pickle
    state_dict = pickle.load(open('/home/roboticslab/yxj/frankapy/data/0408/traj0408.pkl', "rb"))
    q = state_dict[0]["skill_state_dict"]['q']
    print(q.shape)
    dt = 1 / len(q)
    dmp_trajectory = MyDMPTrajectory(dt=dt, num_dims=7, num_basis=42)
    weights = dmp_trajectory.train(q.transpose(1, 0))
    print(weights.shape)

    joint_dmp_info = {'mu': dmp_trajectory.mu_all,
                      'h': dmp_trajectory.h_all,
                      'weights': weights,
    }

    # np.save('/home/roboticslab/yxj/frankapy/cfg/joint_dmp_weights_0414.npy', weights)
    save_path = '/home/roboticslab/yxj/frankapy/data/0414/joint_dmp_weights_0414.pkl'
    with open(save_path, 'wb') as pkl_f:
        pickle.dump(joint_dmp_info, pkl_f, protocol=2)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(time_since_skill_started, q)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])

    y, dy, ddy = dmp_trajectory.execute(tau=1.0)
    plt.subplot(2, 2, 2)
    plt.plot(y)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])

    file_path = '/home/roboticslab/yxj/frankapy/data/0414/joints_memory.txt'
    joints = np.loadtxt(file_path, dtype=float, delimiter=' ')
    print(joints.shape)
    plt.subplot(2, 2, 3)
    plt.plot(joints)
   
    # plt.show()
    # exit(0)
    
    
    

#====================================training

    state_dict = pickle.load( open('/home/roboticslab/yxj/frankapy/data/0408/traj0408.pkl', "rb" ) )
    my_goal = None
    my_y0 = None
    # cartesian_trajectory = {}
    # skill_state_dict = {}

    for key in state_dict.keys():
        skill_dict = state_dict[key]
        
        if skill_dict["skill_description"] == "GuideMode":
            skill_state_dict = skill_dict["skill_state_dict"]
            my_goal = skill_state_dict["q"][-1, :]
            my_y0 = skill_state_dict["q"][0, :]
            # q = skill_state_dict["q"]
            # cartesian_trajectory = process_cartesian_trajectories(skill_state_dict['O_T_EE'], use_quaternions=True, transform_in_row_format=True)
            time_since_skill_started = skill_state_dict["time_since_skill_started"]
    # print(skill_dict)

    # trajectory_start_time = 0.0
    trajectory_end_time = time_since_skill_started[-1]

    # x_range = [trajectory_start_time, trajectory_end_time]
    # trajectory_times = np.reshape(trajectory_start_time+time_since_skill_started,(-1,1))

    with open('/home/roboticslab/yxj/frankapy/data/0414/joint_dmp_weights_0414.pkl', "rb") as pkl_f:
        joint_dmp_info_dict = pickle.load(pkl_f)

    mu, h = joint_dmp_info_dict['mu'], joint_dmp_info_dict['h']
    weights = joint_dmp_info_dict['weights']
    print('joint_dmp_weights: ', weights.shape)

    tau = float(0.5/trajectory_end_time)
    print('tau', tau)

    alpha_y = 60.0
    beta_y = alpha_y/4
    alpha_x = 1.0
    num_basis = 42
    num_dims = 7
    

    # my_dmp = DMPTrajectory(tau,alpha,beta,7,num_basis,num_sensors)
    # joint_dmp_weights, _ = my_dmp.train_using_individual_trajectory('joint', trajectory_times, q)
    # my_dmp.save_weights("joint_dmp_weights_0408.pkl",joint_dmp_weights)

#====================================run

    print('Starting robot')
    fa = FrankaArm()
    
    joint_dmp_info = my_make_joint_dmp_info(tau=tau,
                                            alpha_y=alpha_y,
                                            beta_y=beta_y,
                                            num_dims=num_dims,
                                            num_basis=num_basis,
                                            alpha_x=alpha_x,
                                            mu=mu,
                                            h=h,
                                            weights=weights
                                            )

    # with open(args.joint_dmp_weights_file_path, 'rb') as pkl_f:
    #     joint_dmp_info = pickle.load(pkl_f)
    # if 'initial_sensor_values' in joint_dmp_info:
    #     print('----------')
    #     print('initial_sensor_values: ', joint_dmp_info['initial_sensor_values'])
    #     print('----------')
    # print(joint_dmp_info)

    joint_dmp_info['tau'] = (10 / 7) * (40.7 / 1000) * (100 / 407)

    # initial_sensor_values = np.linspace(1.0, 1.5, 6)
    initial_sensor_values = [1]
    # base_initial_sensor_value = np.array([1, 1.5, 1.5, 1.5, 1, 1.5, 1])
    # base_initial_sensor_value = np.ones(7,)
    initial_joint_error = np.zeros((1, 7))
    joints_memory, pose_memory = [], []
    for iter, current_initial_sensor_value in enumerate(initial_sensor_values):
        print('==============================>')
        print('round: (%d/%d) | init sensor val: %.2f' % (iter, len(initial_sensor_values), current_initial_sensor_value))
        print('Resetting robot to home joints!')
        print('Current goal: ', my_goal)
        print('Psi mu: ', mu)
        print('Psi h: ', h)

        fa.reset_pose(block=True)
        current_home_joints = fa.get_joints()

        print('Home joints: ', current_home_joints)

        print('Resetting robot to home joints!')

        # sensor_values = [1] + [current_initial_sensor_value for i in range(3)] + [1, 1, 1]
        # sensor_values = np.array(sensor_values)

        # my_goal = np.asarray(my_y0) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        """
            DEFAULT_K_GAINS = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
            DEFAULT_D_GAINS = [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0]
        """

        """
            Good params
            k_gains=[600.0, 600.0, 600.0, 600.0, 200.0, 150.0, 50.0],
            d_gains=[50.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0]
        """


        fa.execute_joint_dmp(joint_dmp_info=joint_dmp_info, 
                             duration=(12/1.0), 
                             use_impedance=True,
                             k_gains=[600.0, 600.0, 600.0, 600.0, 200.0, 150.0, 50.0],
                             d_gains=[50.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0],
                             initial_sensor_values=my_goal if isinstance(my_goal, list) else my_goal.reshape(-1).tolist(),
                             block=False)

        import time
        timer = rospy.Rate(50)
        start_time = time.time()
        while True:
            end_time = time.time()
            if (end_time - start_time) >= 12.0:
                break
            joints_memory.append(fa.get_joints().tolist())
            pose_memory.append(fa.get_pose().translation)
            timer.sleep()

        initial_joint_error[iter, :] = fa.get_joints() - current_home_joints
        print('The robot has reached the goal!')
        print('(g - y_0) in current iteration: ', initial_joint_error[iter, :])
    np.save('dmp_joint_error_data.npy', initial_joint_error)
    print('The robot stopped!')
    plt.subplot(2, 2, 4)
    plt.plot(joints_memory)
    plt.show()

    plt.figure()
    plt.plot(pose_memory)
    plt.show()

    # reproduced_file = 'reproduced_traj0407.pkl'
    # with open(reproduced_file, 'wb') as pkl_f:
    #     pickle.dump(goal, pkl_f)
    #     print("Did save skill dict: {}".format(reproduced_file))
