import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys



if __name__ =="__main__":
    # dir = sys.path[0]
    # print(dir)
    # with open(dir+'/../reproduced_traj0407.pkl', 'rb') as pkl_f:
    #     reproduced_traj = pickle.load(pkl_f)
    # print(reproduced_traj)


    # file_path = '/home/roboticslab/yxj/frankapy/cfg/joints_memory.txt'
    # joints = np.loadtxt(file_path, dtype=float, delimiter=' ')
    with open('/home/roboticslab/yxj/frankapy/franka_traj.pkl', 'rb') as pkl_f:
        traj_file = pickle.load(pkl_f)

    joints = traj_file[0]["skill_state_dict"]["q"]
    print(joints.shape)
    plt.figure()
    plt.plot(joints)
    plt.show()


    # joint_error = np.load('./dmp_joint_error_data copy.npy', allow_pickle=True)
    # plt.figure()
    # plt.plot(joint_error)
    # plt.title("joint(0~6)=>(1:1:1:1:1:1:1)")
    # plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    # plt.show()
