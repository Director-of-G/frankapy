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
    joint_error = np.load('./dmp_joint_error_data copy.npy', allow_pickle=True)
    plt.figure()
    plt.plot(joint_error)
    plt.title("joint(0~6)=>(1:1:1:1:1:1:1)")
    plt.legend(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    plt.show()
