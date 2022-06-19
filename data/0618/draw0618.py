import matplotlib.pyplot as plt
import numpy as np
import pickle

pre_traj = '/home/yan/yxj/frankapy/data/0618/'
traj_path = pre_traj+'all_data.pkl'  


with open(traj_path, 'rb') as f1:
    data2 = pickle.load(f1)
    f_list = data2['f_list']
    p_list = data2['p_list']
    kesi_x_list = data2['kesi_x_list']
    pixel_1_list = data2['pixel_1_list']
    pixel_2_list = data2['pixel_2_list']
    time_list = data2['time_list']
    q_and_manipubility_list = data2['q_and_manipubility_list']

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_list, color='b',label = 'f')
    plt.plot(time_list, p_list, color='r',label = 'P')
    plt.title('f and P for vision region')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list,kesi_x_list,label = 'kesi')
    plt.title('kesi_x')
    plt.savefig(pre_traj+'f_P_kesi_x.jpg')

    plt.figure()
    plt.plot(time_list, pixel_1_list,color='b',label = 'vision position')
    plt.plot(time_list, pixel_2_list,color='r',label = 'desired position')
    plt.legend()
    plt.title('vision position vs time')
    plt.savefig(pre_traj+'vision_position.jpg')

    plt.figure()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b',label = 'vision trajectory')
    plt.scatter(pixel_2_list[0][0], pixel_2_list[0][1],color='r',label = 'desired position')
    plt.legend()
    plt.title('vision trajectory')
    plt.savefig(pre_traj+'vision_trajectory.jpg')

    plt.show()




