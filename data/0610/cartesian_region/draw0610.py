import matplotlib.pyplot as plt
import numpy as np
import pickle

pre_traj = '/home/roboticslab/yxj/frankapy/data/0610/cartesian_region_have_joint_sing_2/'
traj_path = pre_traj+'/all_data.pkl'  

with open('/home/roboticslab/yxj/frankapy/data/0610/cartesian_region/all_data.pkl', 'rb') as f1:
    data2 = pickle.load(f1)
    q_m_comp = data2['q_and_manipubility_list']
    t_list_comp = data2['t_list']


with open(traj_path, 'rb') as f:
    data1 = pickle.load(f)

    dist_list = data1['dist']
    q_and_manipubility_list = data1['q_and_manipubility_list']
    pos_list = data1['pos_list']
    quat_list = data1['quat_list']
    p_quat_list = data1['p_quat_list']
    f_quat_list = data1['f_quat_list']
    p_pos_list = data1['p_pos_list']
    f_pos_list = data1['f_pos_list']
    t_list = data1['t_list']

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.plot(t_list,f_quat_list, color='b',label="fo")
    plt.plot(t_list,p_quat_list, color='r',label="Po")
    ax.set_title("f and P for quat")
    ax.legend()
    ax = plt.subplot(2, 2, 2)
    plt.plot(t_list,quat_list)
    ax.set_title("quat")
    ax.legend(['w','x','y','z'])
    ax = plt.subplot(2, 2, 3)
    plt.plot(t_list,f_pos_list, color='b',label="fc")
    plt.plot(t_list,p_pos_list, color='r',label="Pc")
    ax.set_title("f and P for pos")
    ax.legend(['fx','fy','fz','P'])
    ax = plt.subplot(2, 2, 4)
    plt.plot(t_list, pos_list)
    ax.set_title("pos")
    ax.legend(['x','y','z'])
    plt.savefig(pre_traj+'pos_quat.jpg')

    plt.figure()
    for i in range(8):
        ax = plt.subplot(4, 2, i+1)
        plt.plot(t_list, q_and_manipubility_list[:,i])
        if i<7:
            plt.plot(t_list_comp, q_m_comp[:,i])
            ax.legend(['joint %d'%(i+1),'comp'])
        else:
            ax.legend(['manipubility'])
            plt.plot(t_list_comp, q_m_comp[:,i])
            ax.legend(['manipubility','manipubility_comp'])
    plt.savefig(pre_traj+'q_manip.jpg')

    # index = np.argmin(q_and_manipubility_list[:,7])
    # min_manib_joint = q_and_manipubility_list[index,:7]
    # print(min_manib_joint)

    plt.show()





