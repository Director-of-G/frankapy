from tkinter.tix import Tree
import numpy as np
import math
import rospy
import argparse
import pickle
import time
from matplotlib import pyplot as plt
# from autolab_core import RigidTransform, Point
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC

from my_dmp.processing import *
from my_dmp.dmp_class import *


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

def compute_tau_for_franka_interface(tau, expert_traj_len, record_hz, execute_hz=1000):
    """
        compute the parameter tau in 'joint_dmp_info', which will be passed to execute_joint_dmp,
        and further used by franka-interface: joint_dmp_trajectory_generator.cpp
        @params:
            - tau: the ratio of time, which is (T_recording / T_executing)
            - expert_traj_len: the number of points included in expert trajectory
            - record_hz: the number of points included in expert trajectory per second
            - execute_hz: the frequency of franka interface, default is 1000
    """
    return tau * (record_hz / execute_hz) * (100 / expert_traj_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path constants
    parser.add_argument('--traj_path', '-tp', type=str, default='./data/0415/traj_0415.pkl',
                        help='Path of the expert trajectory.')
    parser.add_argument('--memory_path', '-mp', type=str, default='./data/0415/joints_memory.txt',
                        help='Path of real Franka joint angles, recorded by Franka Interface.')
    parser.add_argument('--dmp_weights_path', '-wp', type=str, default='./data/0415/joint_dmp_weights.pkl',
                        help='Path of the calculated dmp weight to save.')

    # frequency constants
    parser.add_argument('--record_frequency', '-rf', type=float, default=10,
                        help='Frequency that the we recorded the expert trajectory, \
                              see my_haptic_subscriber.py for details.')
    parser.add_argument('--execute_frequency', '-ef', type=float, default=1000,
                        help='Frequency that franka interface compute joint angles y, \
                              see franka_interface: joint_dmp_trajectory_generator.cpp for details.')

    # other constants
    parser.add_argument('--run_time', '-rt', type=float, default=40.0)

    args = parser.parse_args()

#====================================plot_trajectories

    # load recorded trajectory
    state_dict = pickle.load(open(args.traj_path, "rb" ) )
    my_goal = None
    my_y0 = None

    for key in state_dict.keys():
        skill_dict = state_dict[key]
        
        if skill_dict["skill_description"] == "GuideMode":
            skill_state_dict = skill_dict["skill_state_dict"]
            expert_trajectory_time = skill_state_dict["time_since_skill_started"]
    
    # get joint trajectory, goal, and initial joint angles
    q = state_dict[0]["skill_state_dict"]['q']
    my_goal = q[-1, :]
    my_y0 = q[0, :]

    # train joint dmp
    dt = 1 / len(q)  # set dt for dmp and cs = (1 / traj_data_points)
    dmp_trajectory = MyDMPTrajectory(dt=dt, num_dims=7, num_basis=42)
    weights = dmp_trajectory.train(q.transpose(1, 0))

    joint_dmp_info = {'mu': dmp_trajectory.mu_all,
                      'h': dmp_trajectory.h_all,
                      'weights': weights,
                     }

    # save joint dmp weights
    save_path = args.dmp_weights_path
    with open(save_path, 'wb') as pkl_f:
        pickle.dump(joint_dmp_info, pkl_f, protocol=2)
    
    plt.figure()
    # plot expert trajectory
    plt.subplot(2, 2, 1)
    plt.plot(expert_trajectory_time, q)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    plt.title('expert trajectory')

    # reproduce trajectory
    tau_reproduce = 1.0
    freqency_scale = args.record_frequency / args.execute_frequency
    time_steps_reproduce = round(dmp_trajectory.cs.time_steps / (tau_reproduce * freqency_scale))
    y, dy, ddy = dmp_trajectory.execute(tau=(tau_reproduce * freqency_scale))
    reproduce_trajectory_time = np.linspace(0,
                                            expert_trajectory_time[-1] / tau_reproduce,
                                            time_steps_reproduce)

    plt.subplot(2, 2, 2)
    plt.plot(reproduce_trajectory_time, y)
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])

#====================================training

    # load dmp weights trained before
    with open(args.dmp_weights_path, "rb") as pkl_f:
        joint_dmp_info_dict = pickle.load(pkl_f)

    mu, h = joint_dmp_info_dict['mu'], joint_dmp_info_dict['h']
    weights = joint_dmp_info_dict['weights']

    tau = 1.0
    tau = compute_tau_for_franka_interface(tau=tau,
                                           expert_traj_len=len(q),
                                           record_hz=args.record_frequency,
                                           execute_hz=args.execute_frequency)

    alpha_y = 60.0
    beta_y = alpha_y / 4
    alpha_x = 1.0
    num_basis = 42
    num_dims = 7

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

    test_rounds = 1

    joints_memory, pose_memory = [], []
    for iter in range(test_rounds):
        print('==============================>')
        print('round: (%d/%d) ' % (iter, test_rounds))
        print('Resetting robot to home joints!')
        print('Current goal: ', my_goal)
        print('Psi mu: ', mu)
        print('Psi h: ', h)

        joints = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, -np.pi / 4]
        fa.goto_joints(joints=joints, block=True)

        current_home_joints = fa.get_joints()
        print('Home joints: ', current_home_joints)
        print('Resetting robot to home joints!')

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
                             duration=args.run_time, 
                             use_impedance=True,
                             k_gains=[600.0, 600.0, 600.0, 600.0, 200.0, 150.0, 50.0],
                             d_gains=[50.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0],
                             initial_sensor_values=my_goal if isinstance(my_goal, list) else my_goal.reshape(-1).tolist(),
                             block=False)

        timer = rospy.Rate(50)
        start_time = time.time()
        while True:
            end_time = time.time()
            if (end_time - start_time) >= args.run_time:
                break
            joints_memory.append(fa.get_joints().tolist())
            pose_memory.append(fa.get_pose().translation)
            timer.sleep()

        print('The robot has reached the goal!')

    joints = np.loadtxt(args.memory_path, dtype=float, delimiter=' ')
    print(joints.shape)
    plt.subplot(2, 2, 3)
    plt.plot(joints)
    
    print('The robot stopped!')
    plt.subplot(2, 2, 4)
    plt.plot(joints_memory)
    plt.show()

    plt.figure()
    plt.plot(pose_memory)
    plt.show()
