import numpy as np
from transformations import euler_matrix

def list_skills(state_dict):
    for key in state_dict.keys():
        skill_dict = state_dict[key]
        print(skill_dict["skill_desc"])

def get_local_ref_frame_rot_mat(trajectory): 
    initial_trans = euler_matrix(trajectory[0, 3], trajectory[0, 4], trajectory[0, 5], 'szyx')
    R0 = initial_trans[:3,:3]
#     R_global_to_cut=np.array([[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,-1.0]])
#     R0=np.matmul(R_global_to_cut,np.transpose(R0)) #works for angledLR as well
    
    return R0