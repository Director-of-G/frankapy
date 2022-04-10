import pickle
import numpy as np
import math
from matplotlib import pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

from scipy.signal import savgol_filter
from transformations import euler_matrix, euler_from_matrix

from my_dmp.utils import get_local_ref_frame_rot_mat

from pyquaternion import Quaternion


def qlog(q):
    vec_len = np.linalg.norm(q.vector)
    if np.linalg.norm(vec_len) <= 1e-6:
        return np.array([0, 0, 0])
    # TODO: Check if there is a difference in ranges of numpy arccos and Eigen acos
    return (q.vector / vec_len ) * np.arccos(q.w)


# NOTE(Mohit): The canonical system is usually just x_j = x(t_j) = exp(-alpha_x/\tau * t_j) (Eq 10 Ude etal.)
def quaternion_phase(t, alpha, goal_t, start_t, int_dt=0.001):
    exec_time = goal_t - start_t
    b = max(1 - alpha * int_dt / exec_time, 1e-8)
    return math.pow(b, (t - start_t) / int_dt)


def quaternion_exp(vec):
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-10:
        return Quaternion(1.0)
    q = np.zeros((4))
    q[1:] = math.sin(vec_norm) * vec / vec_norm
    q[0] = math.cos(vec_norm)
    return Quaternion(q)


class DMPTrajectory(object):
    def __init__(self, tau, alpha, beta, num_dims, num_basis, num_sensors, add_min_jerk=True, alpha_phase=None):
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.alpha_phase = alpha_phase

        self.num_dims = num_dims
        self.num_basis = num_basis
        self.num_sensors = num_sensors

        self._add_min_jerk = add_min_jerk

        self.mean = np.array([np.exp(-i * (0.5 / (self.num_basis - 1))) 
                for i in range(self.num_basis)])
        ## NOTE: This is not the std, but inverse_std, since below we multiply it with (x - mu)**2.
        self.inv_std = [0.5 / (0.65 * (self.mean[i+1]-self.mean[i])**2) 
                for i in range(self.num_basis - 1)]
        self.inv_std += [self.inv_std[-1]]
        self.inv_std = np.array(self.inv_std)
        # Get mu and h for all parameters separately
        self.mu_all = np.zeros((num_dims, num_sensors, num_basis))
        self.h_all = np.zeros((num_dims, num_sensors, num_basis))
        for i in range(num_dims):
            for j in range(num_sensors):
                self.mu_all[i, j] = self.mean
                self.h_all[i, j] = self.inv_std

        self.phi_j = np.ones((self.num_sensors))

        print("Mean: {}".format(np.array_str(self.mean, precision=2,
            suppress_small=True, max_line_width=100)))
        print("Inverse Std: {}".format(np.array_str(self.inv_std, precision=2,
            suppress_small=True, max_line_width=100)))
        # self.print_stuff()

    def print_stuff(self):
        NUM_DIM,  NUM_BASIS = self.num_dims, self.num_basis
        self.c = np.zeros([NUM_DIM, NUM_BASIS])
        self.h = np.zeros([NUM_DIM, NUM_BASIS])
        for i in range(NUM_DIM):
            for j in range(1,NUM_BASIS+1):
                self.c[i,j-1] = np.exp(-(j - 1) * 0.5/(NUM_BASIS - 1))
            for j in range(1,NUM_BASIS):
                self.h[i,j-1] = 0.5 / (0.65 * (self.c[i,j] - self.c[i,j-1])**2)
            self.h[i,NUM_BASIS-1] = self.h[i,NUM_BASIS-2]
        print("Mean: {}".format(np.array_str(self.c[0], precision=2,
            suppress_small=True, max_line_width=100)))
        print("Std: {}".format(np.array_str(self.h[0], precision=2,
            suppress_small=True, max_line_width=100)))

    def get_x_values(self, dt, x_start=1.0):
        x_list = [x_start]
        for _ in range(dt.shape[0]-1):
            dx = -(self.tau * x_list[-1])
            x = x_list[-1] + dx*dt[-1]
            if x < (self.mean[-1] - 3.0*np.sqrt(1.0/self.inv_std[-1])):
                x = 1e-7
            x_list.append(x)
        return np.array(x_list).astype('float64')
    
    def get_quaternion_phase_values(self, start_t, goal_t, dt, canonical_goal_time=None, t_values=None):
        if t_values is None:
            t_values = np.arange(start_t, goal_t+dt, dt)
        canonical_goal_time = goal_t if canonical_goal_time is None else canonical_goal_time
        assert self.alpha_phase is not None
        x = [quaternion_phase(t_values[i], self.alpha_phase, canonical_goal_time, start_t, dt) for i in range(t_values.shape[0])]
        return x
    
    def get_t_values(self, dt, t_start=0.0):
        t_list = [t_start]
        for i in range(dt.shape[0] - 1):
            t_list.append(t_list[-1] + dt[i])
        return np.array(t_list/t_list[-1]).astype('float64')

    def convert_joint_trajectory_to_joint_dmp_training_format(
            self, trajectory_times, trajectory, use_goal_formulation=False):
        
        num_trajectory_points = trajectory.shape[0]
        time = trajectory_times
        dt = time[1:] - time[:-1]
        
        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        
        return self.create_data_using_x(trajectory, time, dt, use_goal_formulation, num_dims=self.num_dims)

    def convert_pose_trajectory_to_pose_dmp_training_format(
            self, trajectory_times, trajectory, local_frame=False, use_goal_formulation=False):

        num_trajectory_points = trajectory.shape[0]
        time = trajectory_times
        dt = time[1:] - time[:-1]

        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        
        ph=np.empty((0,3))      

        ang=None
        
        #####added to allow to change to local reference frame
        if local_frame:       
            print('local_frame',local_frame)    

            R0=get_local_ref_frame_rot_mat(trajectory)
            pos=np.transpose(np.matmul(R0,np.transpose(trajectory[:, :3])))   
            
            for i in range(num_trajectory_points):        
                print('getting localRefFrame data')  
                #print('getting global RefFrame data')
                current_trans = euler_matrix(trajectory[i, 3], trajectory[i, 4], trajectory[i, 5], 'szyx')
                R = current_trans[:3,:3]              
                R_transformed=np.matmul(R0,R)
                
                R4 = trajectory[i,:3].reshape((3,1))
                R_transformed=np.concatenate((R_transformed,R4),axis=1) #3x4  
                R_transformed=np.concatenate((R_transformed,np.array([[0,0,0,1]])),axis=0) #4x4           

                zyx=np.array([(euler_from_matrix(R_transformed, axes='szyx'))])    
             
                ph= np.vstack([ph, zyx]) #tx3 matrix of euler angles
                ang=ph
            pos = np.hstack([pos,ang])
        else: 
            pos = trajectory

        #self.pos_clone_list.append(np.copy(pos))

        return self.create_data_using_x(pos, time, dt, use_goal_formulation, num_dims=self.num_dims)
    
    def convert_pose_trajectory_to_quaternion_dmp_training_format(
            self, trajectory_times, trajectory, local_frame=False, use_goal_formulation=False, dt=0.001):
        '''Use quaternion data format.'''

        time = trajectory_times
        dt_list = time[1:] - time[:-1]

        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        
        #####added to allow to change to local reference frame
        if local_frame:       
            raise ValueError("Not implemented yet.")
        else: 
            pass

        return self.create_data_using_quaternion(trajectory, time, dt_list, delta_t=dt)


    #Calculate Euler angles from raw robot state data
    def convert_pose_trajectory_to_orientation_dmp_training_format(self, 
                                    trajectory_times, trajectory, local_frame=False, use_goal_formulation=False):
        num_trajectory_points = trajectory.shape[0]
        time = trajectory_times
        dt = time[1:] - time[:-1]

        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"

        ph=np.empty((0,3))      

        ang=None
        
        # #########################ORIGINAL
        # #Get rotation matrix for each time point and use to calc 3 EulerAng for each time pt
        if local_frame:
            R0=get_local_ref_frame_rot_mat(trajectory)
            for i in range(num_trajectory_points):        
                print('getting localRefFrame data')  
                #print('getting global RefFrame data')
                current_trans = euler_matrix(trajectory[i, 3], trajectory[i, 4], trajectory[i, 5], 'szyx')
                R = current_trans[:3,:3]              
                R_transformed=np.matmul(R0,R)
                
                R4 = trajectory[i,:3].reshape((3,1))
                R_transformed=np.concatenate((R_transformed,R4),axis=1) #3x4  
                R_transformed=np.concatenate((R_transformed,np.array([[0,0,0,1]])),axis=0) #4x4           

                zyx=np.array([(euler_from_matrix(R_transformed, axes='szyx'))])    
             
                ph= np.vstack([ph, zyx]) #tx3 matrix of euler angles
                ang=ph
        else:
             ang=trajectory[:,3:]

        #self.ang_clone_list.append(np.copy(ang))

        return self.create_data_using_x(ang, time, dt, use_goal_formulation, num_dims=self.num_dims)


    #Calculate Euler angles from raw robot state data
    def convert_pose_trajectory_to_orientation_dmp_training_format(
        self, trajectory_times, trajectory, local_frame=False, use_goal_formulation=False):
        '''Convert data into quaternion DMPs
        '''
        num_trajectory_points = trajectory.shape[0]
        time = trajectory_times
        dt = time[1:] - time[:-1]

        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"

        ph=np.empty((0,3))      

        ang=None
        
        # #########################ORIGINAL
        # #Get rotation matrix for each time point and use to calc 3 EulerAng for each time pt
        if local_frame:
            R0=get_local_ref_frame_rot_mat(trajectory)
            for i in range(num_trajectory_points):        
                print('getting localRefFrame data')  
                #print('getting global RefFrame data')
                current_trans = euler_matrix(trajectory[i, 3], trajectory[i, 4], trajectory[i, 5], 'szyx')
                R = current_trans[:3,:3]              
                R_transformed=np.matmul(R0,R)
                
                R4 = trajectory[i,:3].reshape((3,1))
                R_transformed=np.concatenate((R_transformed,R4),axis=1) #3x4  
                R_transformed=np.concatenate((R_transformed,np.array([[0,0,0,1]])),axis=0) #4x4           

                zyx=np.array([(euler_from_matrix(R_transformed, axes='szyx'))])    
             
                ph= np.vstack([ph, zyx]) #tx3 matrix of euler angles
                ang=ph
        else:
             ang=trajectory[:,3:]

        return self.create_data_using_x(ang, time, dt, use_goal_formulation, num_dims=self.num_dims)

    def convert_pose_trajectory_to_position_dmp_training_format(
            self, trajectory_times, trajectory, local_frame=False, use_goal_formulation=False):

        num_trajectory_points = trajectory.shape[0]
        time = trajectory_times
        dt = time[1:] - time[:-1]

        assert np.min(dt) > 0.0 and np.max(dt) < 1.0, "Recorded time is far off"
        
        #####added to allow to change to local reference frame
        if local_frame:       
            print('local_frame',local_frame)    

            R0=get_local_ref_frame_rot_mat(trajectory)
            pos=np.transpose(np.matmul(R0,np.transpose(trajectory[:, :3])))          
        else: 
            pos = trajectory[:, :3] #uncomment

        #self.pos_clone_list.append(np.copy(pos))

        return self.create_data_using_x(pos, time, dt, use_goal_formulation, num_dims=self.num_dims)

    def quaternion_gradient(self, q, time, dt):
        qd = np.zeros((len(q), 3))
        for i in range(1, len(q)):
            q0, q1 = q[i - 1], q[i]
            #  This check is crucial to ensure that the two quaternions essentially represent the same rotaiton.
            if q0 == q1 or Quaternion.absolute_distance(q0, q1) < 1e-4:
                pass
            else:
                curr_dt = time[i] - time[i - 1]
                qd[i] = 2 * qlog(q1 * q0.conjugate) / curr_dt
        return qd
    
    
    def create_data_using_quaternion(self, x, time, dt, delta_t=0.001):
        q_list = [Quaternion(x[i, 0], x[i, 1], x[i, 2], x[i, 3]) for i in range(x.shape[0])]
        # Normalize the quaternions
        q_list = [q.normalised for q in q_list]
        qd = self.quaternion_gradient(q_list, time, dt)
        for i in range(qd.shape[1]):
            qd[:, i] = savgol_filter(qd[:, i], 51, 3)

        qdd = np.zeros(qd.shape)
        qdd[1:, :] = (qd[1:, :] - qd[:-1, :]) / dt
        for i in range(qdd.shape[1]):
            qdd[:, i] = savgol_filter(qdd[:, i], 51, 3)

        # // Following code is equation (16) from [Ude2014] rearranged to $f_0$
        # for(size_t i = 0; i < R.size(); ++i) {
        #     F.col(i) = t2 * Rdd.col(i)
        #        - (alpha_y * (beta_y * 2 * qLog(R.back() * R[i].conjugate()) - t * Rd.col(i)));
        # }
        t = time[-1, 0] - time[0, 0]
        t2 = t * t
        weight_dims = 3
        force_val = np.zeros((x.shape[0], weight_dims))
        for i in range(x.shape[0]):
            force_val[i] = t2 * qdd[i] - self.alpha * (self.beta * 2 * qlog(q_list[-1] * q_list[i].conjugate) - t * qd[i])
        
        # fig = plt.figure(figsize=(16, 16))
        # ax = fig.add_subplot(131)
        # visualize_Xd_data(time, x, ['W', 'X', 'Y', 'Z'],
        #                   ax=ax, title='Quaternion (q)', xlabel='Time (in sec)')
        # ax = fig.add_subplot(132)
        # visualize_Xd_data(time, qd, ['X', 'Y', 'Z'],
        #                   ax=ax, title='Quaternion velocity (qd)', xlabel='Time (in sec)')
        # ax = fig.add_subplot(133)
        # visualize_Xd_data(time, qdd, ['X', 'Y', 'Z'],
        #                   ax=ax, title='Quaternion acceleration (qdd)', xlabel='Time (in sec)')
        # plt.show()
        
        # Get the x values
        x_arr_quat = self.get_quaternion_phase_values(time[0, 0], time[-1, 0], delta_t, 
                                                      canonical_goal_time=time[-1, 0])
        x_arr_quat = np.array(x_arr_quat)[:x.shape[0]]
        x_arr = x_arr_quat

        # x_arr = self.get_x_values(dt)
        # Repeat the last x to get equal length array. Shape (T)
        # x_arr = np.hstack([x_arr, x_arr[-1:]])

        # fig = plt.figure(figsize=(16, 16))
        # ax = fig.add_subplot(111)
        # ax.plot(np.arange(x.shape[0]), x_arr, c='r')
        # ax.plot(np.arange(x.shape[0]), x_arr_quat, c='b')
        # plt.show()

        # x_arr will be of shape (T, N, M, K)
        for _, axis_size in enumerate([weight_dims, self.num_sensors, self.num_basis]):
            x_arr = np.repeat(x_arr[..., None], axis_size, axis=-1)
        assert x_arr.shape == (time.shape[0], weight_dims, self.num_sensors, self.num_basis)

        psi_tijk = np.exp(-self.h_all * (x_arr - self.mu_all)**2)
        psi_tij_sum = np.sum(psi_tijk, axis=-1, keepdims=True)
        feat_tijk = (psi_tijk * x_arr) / (psi_tij_sum + 1e-10)

        # This reshape happens according to "C" order, i.e., last axis change first
        X = feat_tijk.copy().reshape(x_arr.shape[0], weight_dims, -1)
        y = force_val.copy()
        assert X.shape[0] == y.shape[0], "X, y n_samples do not match"
        assert X.shape[1] == y.shape[1], "X, y n_dims do not match"

        # X^T\beta = y (where we have to find \beta)

        return {'X': X, 'y': y}


    def create_data_using_x(self, x, time, dt, use_goal_formulation, num_dims=None):
        assert num_dims is not None, "Invalid number of dims used"

        # Get velocity and acceleration.
        d_x = (x[1:, :] - x[:-1, :]) / dt

        for i in range(d_x.shape[1]):
            d_x[:, i] = savgol_filter(d_x[:, i], 101, 5)

        # Repeat the last x to get equal length array.
        d_x = np.vstack([d_x, d_x[-1:, :]])
        dd_x = (d_x[1:, :] - d_x[:-1, :]) / dt

        # Repeat the last x to get equal length array.
        dd_x = np.vstack([dd_x, dd_x[-1:, :]])
        #print("dd_x: max: {:.3f}, min: {:.3f}, mean: {:.3f}".format(
        #    dd_x.min(), dd_x.max(), np.mean(dd_x)))
        #self.dx_clone_list.append(d_x.copy())
        #self.ddx_clone_list.append(dd_x.copy())
        
        # fig = plt.figure(figsize=(16, 16))
        # ax = fig.add_subplot(131)
        # visualize_Xd_data(time, x, ['X', 'Y', 'Z"'], ax=ax, title='Position (x)', xlabel='Time (in sec)')
        # ax = fig.add_subplot(132)
        # visualize_Xd_data(time, d_x, ['vX', 'Y', 'vZ'], ax=ax, title='velocity (dx)', xlabel='Time (in sec)')
        # ax = fig.add_subplot(133)
        # visualize_Xd_data(time, dd_x, ['X', 'Y', 'Z'], ax=ax, title='acceleration (ddx)', xlabel='Time (in sec)')
        # plt.show()
        
        y0 = x[0]

        force_val = dd_x/(self.tau**2) - self.alpha*(self.beta*(y0-x) - d_x/self.tau)
        force_val = force_val/(self.alpha*self.beta)
        # Get the x values
        x_arr = self.get_x_values(dt)
        # Repeat the last x to get equal length array. Shape (T)
        x_arr = np.hstack([x_arr, x_arr[-1:]])

        # x_arr will be of shape (T, N, M, K)
        for _, axis_size in enumerate(
                [num_dims, self.num_sensors, self.num_basis]):
            x_arr = np.repeat(x_arr[..., None], axis_size, axis=-1)
        assert x_arr.shape == \
          (time.shape[0], num_dims, self.num_sensors, self.num_basis)

        psi_tijk = np.exp(-self.h_all * (x_arr - self.mu_all)**2)
        psi_tij_sum = np.sum(psi_tijk, axis=-1, keepdims=True)

        feat_tijk = (psi_tijk * x_arr) / (psi_tij_sum + 1e-10)

        # Get the minimum jerk features
        min_jerk_t = np.minimum(-np.log(x_arr[:,:,:,0:1])*2,
                                np.ones(x_arr[:,:,:,0:1].shape))
        feat_min_jerk_tijk = (min_jerk_t**3)*(6*(min_jerk_t**2)-15*min_jerk_t+10)

        feat_tijk = np.concatenate([feat_min_jerk_tijk, feat_tijk], axis=-1)

        # use displace (g - y0) as phi_j where j = 0
        if use_goal_formulation:
            # calculate (g - y0)
            f_shape = feat_tijk.shape
            x_delta = x[-1, :] - x[0, :]
            phi_ij = np.ones((f_shape[1], f_shape[2]))
            # phi_ij[-1, 0] = x_delta[-1]
            phi_ij[:, 0] = x_delta
            phi_ijk = np.tile(phi_ij[:, :, np.newaxis], (1, 1, f_shape[3]))
            phi_tijk = np.tile(phi_ijk[np.newaxis, ...], (f_shape[0], 1, 1, 1))
            feat_tijk = feat_tijk * phi_tijk

        # This reshape happens according to "C" order, i.e., last axis change
        # first, this means that 1400 parameters are laid according to each
        # dim.
        X = feat_tijk.copy().reshape(x_arr.shape[0], num_dims, -1)
        y = force_val.copy()
        assert X.shape[0] == y.shape[0], "X, y n_samples do not match"
        assert X.shape[1] == y.shape[1], "X, y n_dims do not match"

        # X^T\beta = y (where we have to find \beta)

        return {'X': X, 'y': y}
    

    def run_quaternion_dmp_with_weights(self, weights, q0, qgoal, dt, traj_time=100,  phi_j=None, quat_canonical_goal_time=None):
        '''Run DMP with given weights.
        weights: array of weights. size: (N*M*K, 1) i.e. 
            (num_dims*num_sensors*num_basis, 1)
        y0: Start location for dmps. Array of size (N,)
        dt: Time step to use. Float.
        traj_time: Time length to sample trajectories. Integer
        '''
        x = 1.0
        q  = np.zeros((traj_time, 4))
        qd = np.zeros((traj_time, 3))
        qdd = np.zeros((traj_time, 3))
        q[0] = q0
        x_log = []

        start_time = 0
        times = [start_time + dt * i for i in range(traj_time)]

        goal_quat = Quaternion(qgoal[0], qgoal[1], qgoal[2], qgoal[3])
        # NOTE: quat_canonical_goal_time is used to set the canonical variable to 0 
        # which is necessary to drive the system to a stable state.
        if quat_canonical_goal_time is None:
            quat_canonical_goal_time = times[-1]
        # else:
        #     assert times[-1] > quat_canonical_goal_time, "Trajectory duration should be larger than canonical system being set to 0."

        for i in range(1, traj_time):
            next_q_quat, qd_val, qdd_val, x = self.quaternion_dmp_step(
                times[i], times[0], quat_canonical_goal_time,
                goal_quat,
                q[i - 1],
                qd[i - 1],
                qdd[i - 1],
                weights,
                dt)
            qd[i] = qd_val
            qdd[i] = qdd_val
            q[i][0] = next_q_quat.w
            q[i][1:] = next_q_quat.vector
            x_log.append(x)

        return q, qd, qdd, x_log


    def quaternion_dmp_step(self, t, start_t, goal_t, goal_quat,  last_q, last_qd, last_qdd, weights, dt):
        '''
        t:       double. Current time in trajectory.
        start_t: double. Start time for trajectory.
        goal_t:  double. Goal time to reach trajectory.
        goal_quat: Quaternion. Goal quaternion to reach.
        last_q:   Array (4). Quaternion in array format (w, x, y, z)
        last_qd:  Array (3). Quaternion velocity at last step.
        last_qdd: Array (3). Quaternion acceleration at last step.
        weights:  Array. Learned DMP weights
        dt:      double. Integratoin time.  
        '''
        x = quaternion_phase(t, self.alpha_phase, goal_t, start_t, int_dt=dt)
        if t > goal_t:
            x = 0

        last_q_quat = Quaternion(last_q)

        # Calculate the RBF activations (maybe make this a function)
        psi_ijk = np.exp(-self.h_all * (x - self.mu_all) ** 2)
        psi_ij_sum = np.sum(psi_ijk, axis=2, keepdims=True)

        # Calculate the forcing term (maybe make this a function)
        f = (psi_ijk * weights * x).sum(axis=2, keepdims=True) / (psi_ij_sum + 1e-10)

        exec_time = goal_t - start_t
        exec_time_squared = exec_time * exec_time

        f = f.squeeze()
        assert len(f.shape) == 1, "Forcing function should only have one sensor v alue for now"
        qdd = (self.alpha * (self.beta * 2.0 * qlog(goal_quat * last_q_quat.conjugate) - exec_time * last_qd) + f) / exec_time_squared

        # Take a step for the canonical system, Eq (24) in Ude et al.
        next_q_quat = quaternion_exp(dt / 2.0 * last_qd) * last_q_quat

        qd = last_qd + dt * qdd

        return next_q_quat, qd, qdd, x


    def run_dmp_with_weights(self, weights, y0, dt, traj_time=100,  phi_j=None):
        '''Run DMP with given weights.
        weights: array of weights. size: (N*M*K, 1) i.e. 
            (num_dims*num_sensors*num_basis, 1)
        y0: Start location for dmps. Array of size (N,)
        dt: Time step to use. Float.
        traj_time: Time length to sample trajectories. Integer
        '''
        x = 1.0
        y  = np.zeros((traj_time, self.num_dims))
        dy = np.zeros((traj_time, self.num_dims))
        y[0] = y0
        # This reshape happens along the vector i.e. the first (M*K) values 
        # belong to dimension 0 (i.e. N = 0). Following (M*K) values belong to
        # dimension 1 (i.e. N = 1), and so forth.
        # NOTE: We add 1 for the weights of the jerk basis function

        min_jerk_arr = np.zeros((traj_time))

        x_log = []
        psi_log = []
        min_jerk_log = []
        for i in range(traj_time - 1):
            # psi_ijk is of shape (N, M, K)
            psi_ijk = np.exp(-self.h_all * (x-self.mu_all)**2)

            psi_ij_sum = np.sum(psi_ijk, axis=2, keepdims=True)

            f = (psi_ijk * weights[:, :, 1:] * x).sum(
                    axis=2, keepdims=True) / (psi_ij_sum + 1e-10)
            # f_min_jerk = (i * dt)/(traj_time * dt)
            f_min_jerk = min(-np.log(x)*2, 1)
            f_min_jerk = (f_min_jerk**3)*(6*(f_min_jerk**2) - 15*f_min_jerk+ 10)
            psi_ij_jerk = weights[:, :, 0:1] * f_min_jerk

            # for debug
            min_jerk_arr[i] = f_min_jerk

            # calculate f(x; w_j)l -- shape (N, M)
            all_f_ij = self.alpha * self.beta * (f + psi_ij_jerk).squeeze()

            # Calculate sum_j(phi_j * f(x; w_j) -- shape (N,)
            
            if phi_j is None:
                phi_j = self.phi_j

            #ORIGINAL CODE
            # if len(phi_j.shape) == 1:
            #     all_f_i = np.dot(all_f_ij, phi_j) #Uncomment this if num sensors =2

            #     #all_f_i = np.dot((self.alpha * self.beta * (f + psi_ij_jerk)), phi_j) #comment out if num sensors=2 (updated to make matrix dims work for num sensors=1)
            #     #all_f_i=all_f_i.squeeze() #comment out if num sensors=2
            
            # elif len(phi_j.shape) == 2:
            #     all_f_i = np.sum(all_f_ij * phi_j, axis=1)
            # else:
            #     raise ValueError("Incorrect shape for phi_j")

            #10/31 update to fix issue and make it work if num sensors is 1 or 2:
            if phi_j.shape == (1,):
                #all_f_i = np.dot(all_f_ij, phi_j) #Uncomment this if num sensors =2

                all_f_i = np.dot((self.alpha * self.beta * (f + psi_ij_jerk)), phi_j) #comment out if num sensors=2 (updated to make matrix dims work for num sensors=1)
                all_f_i=all_f_i.squeeze() #comment out if num sensors=2
            
            elif phi_j.shape == (2,):
                #all_f_i = np.sum(all_f_ij * phi_j, axis=1)
                all_f_i = np.dot(all_f_ij, phi_j) #Uncomment this if num sensors =2
            else:
                raise ValueError("Incorrect shape for phi_j")
            ###end 10/31 update

            
            ddy = self.alpha*(self.beta*(y0 - y[i]) - dy[i]/self.tau) + all_f_i
            ddy = ddy * (self.tau ** 2)
            dy[i+1] = dy[i] + ddy * dt
            y[i+1] = y[i] + dy[i+1] * dt

            x_log.append(x)
            psi_log.append(psi_ijk)
            min_jerk_log.append(f_min_jerk)

            x += ((-self.tau * x) * dt)
            if (x < self.mean[-1] - 3.0*np.sqrt(1.0/(self.inv_std[-1]))):
                x = 1e-7

        return y, dy, x_log, np.array(psi_log), np.array(min_jerk_log)


    def train(self, X_train, y_train, X_test, y_test, use_ridge=False,
              fit_intercept=True):
        reg_lambda = 0.01
        if use_ridge:
            clf = Ridge(alpha=reg_lambda,
                        fit_intercept=fit_intercept).fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],
                          fit_intercept=fit_intercept).fit(X_train, y_train)
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_train)
        print("Score (max 1.0) Train: {:.3f}, Test: {:.3f}".format(
            train_score, test_score))

        M = np.linalg.inv(np.dot(X_train.T, X_train) + 
                          reg_lambda*np.eye(X_train.shape[1]))
        N = np.dot(X_train.T, y_train)
        weights = np.dot(M, N)
        ridge_weights = clf.coef_
        preds = np.dot(X_train, weights)
        self.train_clf_preds = y_pred.copy()
        self.train_preds = preds.copy()
        self.train_gt = y_train.copy()
        print("Error in pred: {:.6f}".format(
            np.linalg.norm(preds - y_train)/y_train.shape[0]))

        return clf
    
    def ridge_regression(self, X, y, dmp_type):
        train_size = int(X.shape[0] * 1.0)
        print("Train size: {} Test size: {}".format(train_size, X.shape[0]-train_size))

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[:train_size], y[:train_size]

        # Train a classifier separately for each dimension.
        weights = []
        for i in range(self.num_dims):
            clf = self.train(X_train[:, i, :],
                             y_train[:, i],
                             X_test[:, i],
                             y_test[:, i], 
                             use_ridge=True,
                             fit_intercept=False)
            weights.append(clf.coef_.copy().squeeze())
            print("Got weights for dim: {}, min: {:.3f}, max: {:.3f}, avg: {:.3f}".
                    format(i, weights[-1].min(),
                           weights[-1].max(), weights[-1].mean()))

        basis = self.num_basis + 1 if dmp_type != 'quaternion' else self.num_basis
        weights_ijk = np.array([np.reshape(W, (self.num_sensors, basis)) for W in weights])
        return weights_ijk


    def train_using_individual_trajectory(self, dmp_type, trajectory_times, trajectory, 
                                          local_frame=False, use_goal_formulation=False, **kwargs):
        X, y = [], []

        if dmp_type == 'pose':
            data = self.convert_pose_trajectory_to_pose_dmp_training_format(
                trajectory_times, trajectory, local_frame=local_frame,
                use_goal_formulation=use_goal_formulation, **kwargs)
        elif dmp_type == 'position':
            data = self.convert_pose_trajectory_to_position_dmp_training_format(
                trajectory_times, trajectory, local_frame=local_frame,
                use_goal_formulation=use_goal_formulation, **kwargs)
        elif dmp_type == 'orientation':
            data = self.convert_pose_trajectory_to_orientation_dmp_training_format(
                trajectory_times, trajectory, local_frame=local_frame,
                use_goal_formulation=use_goal_formulation, **kwargs)
        elif dmp_type == 'quaternion':
            data = self.convert_pose_trajectory_to_quaternion_dmp_training_format(
                trajectory_times, trajectory, local_frame=local_frame,
                use_goal_formulation=use_goal_formulation, **kwargs)
        else:
            data = self.convert_joint_trajectory_to_joint_dmp_training_format(
                trajectory_times, trajectory, use_goal_formulation=use_goal_formulation, **kwargs)

        assert (type(data['X']) is np.ndarray
                and type(data['y']) is np.ndarray), "Incorrect data type returned"

        X.append(data['X'])
        y.append(data['y'])
        X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)

        weights_ijk = self.ridge_regression(X, y, dmp_type)
        return weights_ijk, {'X': X, 'y': y}


    def save_weights(self, save_path, weights, **kwargs):
        if self._add_min_jerk:
            mean = [0.0] + self.mean.tolist()
            inv_std = [0.0] + self.inv_std.tolist()
        else:
            mean = self.mean.tolist()
            inv_std = self.inv_std.tolist()
        phi_j_list = self.phi_j.tolist()
        weights_list = weights.tolist()

        num_basis = self.num_basis + 1 if self._add_min_jerk else self.num_basis

        data_dict = {
            'tau': self.tau,
            'alpha': self.alpha,
            'beta': self.beta,
            'num_dims': self.num_dims,
            'num_basis': num_basis,
            'num_sensors': self.num_sensors,
            'mu': mean,
            'h': inv_std,
            'phi_j': phi_j_list,
            'weights': weights_list,
        }
        for k, v in kwargs.items():
            data_dict[k] = v

        with open(save_path, 'wb') as pkl_f:
            pickle.dump(data_dict, pkl_f, protocol=2)
            print("Did save dmp params: {}".format(save_path))

    def get_dmp_params(self, weights, **kwargs):
        if self._add_min_jerk:
            mean = [0.0] + self.mean.tolist()
            inv_std = [0.0] + self.inv_std.tolist()
        else:
            mean = self.mean.tolist()
            inv_std = self.inv_std.tolist()
        phi_j_list = self.phi_j.tolist()
        print(weights.shape)
        weights_list = np.array(weights).flatten().tolist()

        num_basis = self.num_basis + 1 if self._add_min_jerk else self.num_basis

        data_dict = {
            'tau': self.tau,
            'alpha': self.alpha,
            'beta': self.beta,
            'num_dims': self.num_dims,
            'num_basis': num_basis,
            'num_sensors': self.num_sensors,
            'mu': mean,
            'h': inv_std,
            'phi_j': phi_j_list,
            'weights': weights_list,
        }
        for k, v in kwargs.items():
            data_dict[k] = v

        return data_dict

    def save_weights_csv(self,filename,weights):
        data=weights
        print('shape weights is ', data.shape)
        print(data)

        with open(filename, 'w') as outfile:

            #outfile.write('# Array shape: {0}\n'.format(data.shape))

            #for data_slice in data:  #produces slices for each x, y, z dimension
            for i in range(0,data.shape[0]):    
                #print(data_slice)
                print(i)
                data_slice=data[i,:,:]
                print(data[i,:,:])

                np.savetxt(outfile, data_slice, fmt='%-10.8f', delimiter=' ')               


                #outfile.write('# New slice\n')
            #a=open(filename,'r')
            #print('file contains')
            #print(a.read())

        #print("Saved weights to csv")
