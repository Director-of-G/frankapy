import numpy as np
from transformations import euler_from_matrix, quaternion_from_matrix

from pyquaternion import Quaternion


def process_cartesian_trajectories(trajectory, axes='sxyz', use_quaternions=False, transform_in_row_format=True):
    
    num_trajectory_points = trajectory.shape[0]
    if use_quaternions:
        cartesian_trajectory = np.zeros((num_trajectory_points, 7))
    else:
        cartesian_trajectory = np.zeros((num_trajectory_points, 6))
    if transform_in_row_format:
        transformation_trajectory = trajectory.reshape(-1, 4, 4)
        cartesian_trajectory[:,:3] = transformation_trajectory[:, :3, 3]
        transformation_trajectory[:, :3, 3] = 0
    else:
        cartesian_trajectory[:,:3] = trajectory[:,12:15]
        
        transformation_trajectory_1 = trajectory[:,:4].reshape((-1,4,1))
        transformation_trajectory_2 = trajectory[:,4:8].reshape((-1,4,1))
        transformation_trajectory_3 = trajectory[:,8:12].reshape((-1,4,1))
        transformation_trajectory_4 = trajectory[:,12:].reshape((-1,4,1))
        transformation_trajectory = np.concatenate([transformation_trajectory_1, transformation_trajectory_2,
                                                    transformation_trajectory_3, transformation_trajectory_4], axis=2)
    
    assert np.all(transformation_trajectory[:, 3, :3] == 0) and np.all(transformation_trajectory[:, 3, 3] == 1), \
        "Invalid transformations in trajectory."

    quaternion_dists = []
    for i in range(num_trajectory_points):
        if use_quaternions:
            curr_q_arr = quaternion_from_matrix(transformation_trajectory[i, ...])
            if i > 0:
                if np.all(np.abs(curr_q_arr - cartesian_trajectory[i - 1, 3:]) < 1e-6):
                    pass
                else:
                    # Since q and -q quaternions represent the same orientation we want to choose the quaternion which is closer
                    last_q = Quaternion(cartesian_trajectory[i - 1, 3:])
                    curr_q = Quaternion(curr_q_arr)
                    same_q_opposite = Quaternion(-curr_q)
                    # Avoid flipping to ensure that quaternions do not flip if the dists are almost very similar.
                    if Quaternion.distance(last_q, same_q_opposite) < Quaternion.distance(last_q, curr_q) + 1e-6:
                        curr_q_arr = -curr_q_arr

                # Verify that the last quaternion and the current quaternion are closer 
                quaternion_dists.append(Quaternion.distance(Quaternion(cartesian_trajectory[i-1, 3:]), 
                                                            Quaternion(curr_q_arr)))

            cartesian_trajectory[i, 3:] = curr_q_arr
        else:
            cartesian_trajectory[i, 3:] = euler_from_matrix(transformation_trajectory[i,:,:], axes=axes)
            # Don't remember exactly why we did this but most likely we will be changing things in the future.
            cartesian_trajectory[i, 3:] *= -1

    # Makes euler angles continuous
    if not use_quaternions:
        for i in range(3,6):
            cartesian_trajectory[:,i] = np.unwrap(cartesian_trajectory[:,i])
    else:
        # TODO: We should probably normalize quaternions
        pass

    return cartesian_trajectory

def get_start_and_end_times(trajectory_times, cartesian_trajectory, joint_trajectory, threshold):
    
    num_trajectory_points = trajectory_times.shape[0]
    start_trajectory_index = 0
    end_trajectory_index = num_trajectory_points
        
    previous_robot_pose = cartesian_trajectory[0,:]
    for i in range(1,num_trajectory_points):
        current_robot_pose = cartesian_trajectory[i,:]
        if(np.linalg.norm(current_robot_pose - previous_robot_pose) > threshold):
            start_trajectory_index = i - 1
            break
        previous_robot_pose = current_robot_pose

                
    previous_robot_pose = cartesian_trajectory[-1,:]
    for i in reversed(range(0,num_trajectory_points-1)):
        current_robot_pose = cartesian_trajectory[i,:]
        if(np.linalg.norm(current_robot_pose - previous_robot_pose) > threshold):
            end_trajectory_index = i + 2
            break

    return trajectory_times[start_trajectory_index], trajectory_times[end_trajectory_index]


def truncate_trajectory(trajectory_times, cartesian_trajectory, joint_trajectory, start_time, end_time):
    
    num_trajectory_points = trajectory_times.shape[0]
    start_trajectory_index = 0
    end_trajectory_index = num_trajectory_points
        
    for i in range(0,num_trajectory_points):
        if(start_time - trajectory_times[i]) <= 0:
            start_trajectory_index = i 
            break

    for i in reversed(range(0,num_trajectory_points-1)):
        if(end_time - trajectory_times[i]) >= 0:
            end_trajectory_index = i
            break

    truncated_trajectory_times = trajectory_times[start_trajectory_index:end_trajectory_index]
    truncated_trajectory_times -= truncated_trajectory_times[0]
    truncated_cartesian_trajectory = cartesian_trajectory[start_trajectory_index:end_trajectory_index,:]
    truncated_joint_trajectory = joint_trajectory[start_trajectory_index:end_trajectory_index,:]
        
    return truncated_trajectory_times, truncated_cartesian_trajectory, truncated_joint_trajectory