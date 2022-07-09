import struct
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt, units
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
print(sys.path)
from my_utils import Quat

from my_gzb_adaptive_control import CartesianSpaceRegion, CartesianQuatSpaceRegion, ImageSpaceRegion
from torch.utils.tensorboard import SummaryWriter

import pdb

class MyConstants(object):
    FX_HAT = 3759.66467
    FY_HAT = 3759.66467
    U0 = 960.5
    V0 = 540.5
    IMG_H = 1080
    IMG_W = 1920
    CAM_HEIGHT = 1.50

class data_collection(object):  # to get Franka data from Gazebo
    def __init__(self) -> None:
        self.J = np.zeros((6, 7))
        self.q = np.zeros((7,))
        self.trans = np.zeros((3,))
        self.quat = np.zeros((4,))
        self.J_ready = False
        self.q_ready = False
        self.pose_ready = False
        self.vision_ready = False
        self.x = np.array([0,0])

    def zero_jacobian_callback(self, msg):
        self.J = np.array(msg.data).reshape(6, 7)
        if not self.J_ready:
            self.J_ready = True

    def joint_angles_callback(self, msg):
        self.q = np.array(msg.data).reshape(7,)
        if not self.q_ready:
            self.q_ready = True

    def pose_callback(self, msg):
        self.trans = np.array(msg.data)[:3].reshape(3,)
        self.quat = np.array(msg.data)[[6, 3, 4, 5]].reshape(4,)
        if not self.pose_ready:
            self.pose_ready = True

    def vision_callback(self, msg):
        self.x = np.array([msg.point.x,msg.point.y])
        if not self.vision_ready:
            self.vision_ready = True

    def is_data_without_vision_ready(self):
        return self.J_ready & self.q_ready & self.pose_ready

    def is_data_with_vision_ready(self):
        return self.J_ready & self.q_ready & self.pose_ready & self.vision_ready

class FrankaInfoStruct(object):
    def __init__(self) -> None:
        self.x = np.array([0, 0])
        self.quat = np.array([-0.17492908847362298, \
                               0.6884405719242297, \
                               0.6818253503208791, \
                               0.17479727175084528])  # (w, x, y, z)
        self.trans = np.array([-1, -1, 0.5])  # (deprecated, deprecated, z)

def get_Js_hat(data_c):
    x = data_c.x.reshape(-1,)
    ee_pose_quat = data_c.quat[[1, 2, 3, 0]]
    ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
    p_s_in_panda_EE = np.array([0.058690, 0.067458, -0.053400])
    p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)

    Z = MyConstants.CAM_HEIGHT - data_c.trans.reshape(-1,)[2]

    R_b2c = np.array([[-1, 0,  0],
                      [0,  1,  0],
                      [0,  0, -1]])
    Js = np.array([[MyConstants.FX_HAT / Z, 0,    - (x[0] - MyConstants.U0) / Z, 0, 0, 0], \
                    [0,    MyConstants.FY_HAT / Z, - (x[1] - MyConstants.V0) / Z, 0, 0, 0]])
    cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                          [-p_s[2],  0,       p_s[0]],
                          [p_s[1],  -p_s[0],  0]])
    Jrot = np.block([[R_b2c, R_b2c], \
                     [np.zeros((3, 6))]])
    Jvel = np.block([[np.eye(3), np.zeros((3, 3))], \
                     [np.zeros((3, 3)), cross_mat]])

    return (Js @ Jrot @ Jvel).reshape(2, 6)

def test_zero_Jacobian():
    class data_collection(object):
        def __init__(self) -> None:
            self.J = np.zeros((6, 7))
            self.J_ready = False

        def zero_jacobian_callback(self, msg):
            self.J = np.array(msg.data).reshape(6, 7)
            if not self.J_ready:
                self.J_ready = True

    data_c = data_collection()
        
    nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    rate_ = rospy.Rate(100)
    
    while not data_c.J_ready:
        pass

    start_time = time.time()

    while not rospy.is_shutdown():
        body_twist = np.array([0, 0, -0.1, 0, 0, 0])
        J = data_c.J
        J_pinv = J.T @ np.linalg.inv(J @ J.T)
        dq_d = J_pinv @ body_twist.reshape(-1, 1)
        msg = Float64MultiArray()
        msg.data = dq_d.reshape(7,)
        pub_.publish(msg)                     
        rate_.sleep()

        if time.time() - start_time >= 5.0:
            msg = Float64MultiArray()
            msg.data = np.zeros(7,).reshape(7,)
            pub_.publish(msg)
            break

def test_plot_3D(range:np.ndarray=np.array([[-0.35 - 0.5, 0.30 - 0.5], [0.25, 0.65], [0.40, 0.70]])):
    # plot Cartesian region
    
    x, y, z = np.indices((2, 2, 2))
    x, y, z = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)

    zeros_mask, ones_mask = (x == 0), (x == 1)
    x[zeros_mask], x[ones_mask] = range[0, 0], range[0, 1]

    zeros_mask, ones_mask = (y == 0), (y == 1)
    y[zeros_mask], y[ones_mask] = range[1, 0], range[1, 1]

    zeros_mask, ones_mask = (z == 0), (z == 1)
    z[zeros_mask], z[ones_mask] = range[2, 0], range[2, 1]

    filled = np.ones((1, 1, 1))
    cFace = np.where(filled, '#00AAAAA0', '#00AAAAA0')
    cEdge = np.where(filled, '#008888', '#008888')
    ax = plt.subplot(projection='3d')
    ax.voxels(x, y, z, filled=filled, facecolors=cFace, edgecolors=cEdge)

    # plot FOV
    verts2 = [(-0.5 + 0.2758, 0.5 - 0.1551, 0.419823), 
              (-0.5 - 0.2758, 0.5 - 0.1551, 0.419823), 
              (-0.5 - 0.2758, 0.5 + 0.1551, 0.419823), 
              (-0.5 + 0.2758, 0.5 + 0.1551, 0.419823), 
              (-0.5, 0.5, 1.5)]
    faces2 = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4], [0, 1, 2, 3]]
    
    poly3d2 = [[verts2[vert_id] for vert_id in face] for face in faces2]
    
    x2, y2, z2 = zip(*verts2)
    ax.scatter(x2, y2, z2)
    
    collection2 = Poly3DCollection(poly3d2, edgecolors= 'r', facecolor= [0.5, 0.5, 1], linewidths=1, alpha=0.3)
    ax.add_collection3d(collection2)

    plt.xlim((-0.8, 0.4))
    plt.ylim((0.2, 0.7))
    ax.set_zlim(0.3, 1.5)
    plt.gca().set_box_aspect((12, 5, 12))

    plt.show()

def test_Image_Jacobian():
    # Brief: define necessary classes

    class traj_generator(object):
        def __init__(self, dir:np.ndarray=None, vel:np.ndarray=None, bound=200) -> None:
            self.dir = dir
            self.vel = vel
            self.bound = bound

            # FSM
            # 0 ==> stationary
            # 1 ==> positive direction
            # 2 ==> negative direction
            # 3 ==> transfer after positive direction
            # 4 ==> transfer after negative direction
            self.state = 0
            self.last_vel = np.zeros((2,))

        def in_region(self, x):
            if (x[0] < self.bound or x[0] > MyConstants.IMG_W - self.bound) or \
               (x[1] < self.bound or x[1] > MyConstants.IMG_H - self.bound):
                return False

            else:
                return True

        def get_velocity(self, x):
            x = x.reshape(-1,)
            print('x: ', x)
            if self.state == 0:
                return np.zeros((2,))
            elif self.state == 1:
                if not self.in_region(x):
                    self.state = 3
                    print('Switching to state 3!')
                    return - self.vel * self.dir
                else:
                    return self.vel * self.dir
            elif self.state == 2:
                if not self.in_region(x):
                    self.state = 4
                    print('Switching to state 4!')
                    return self.vel * self.dir
                else:
                    return - self.vel * self.dir
            elif self.state == 3:
                if self.in_region(x):
                    self.state = 2
                    print('Switching to state 2!')
                return - self.vel * self.dir
            elif self.state == 4:
                if self.in_region(x):
                    self.state = 1
                    print('Switching to state 1!')
                return self.vel * self.dir
            else:
                return np.zeros((2,))

    class velocity_calculator(object):
        def __init__(self) -> None:
            self.vels = np.zeros((3, 2)).tolist()
        
        def get_vel(self, x):
            x = x.reshape(-1,).tolist()
            self.vels.pop(0)
            self.vels.append(x)
            
            vels_array = np.array(self.vels)
            return (vels_array[2] - vels_array[0]) / (2 * (1 / 30))

    nh_ = rospy.init_node('image_jacobian_testnode', anonymous=True)

    rate = rospy.Rate(30)
    data_c = data_collection()
    traj_g = traj_generator(dir=np.array([1, 0]), vel=0.05, bound=200)
    vel_c = velocity_calculator()
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    sub_q_ = rospy.Subscriber('/gazebo_sim/joint_angles', Float64MultiArray, data_c.joint_angles_callback, queue_size=1)
    sub_ee_pose_ = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, data_c.pose_callback, queue_size=1)
    sub_vision_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_callback, queue_size=1)
    
    image_space_region = ImageSpaceRegion(b=np.array([1920, 1080]))
    cartesian_space_region = CartesianSpaceRegion()
    cartesian_quat_space_region = CartesianQuatSpaceRegion()

    cartesian_space_region.set_r_c(np.array([-0.0711823860573642849, 0.48430624374805804, 0.669872105919327]))
    cartesian_space_region.set_c(np.array([0.05, 0.05, 0.05]).reshape(1, 3))
    cartesian_space_region.set_Kc(np.array([5e-5, 5e-5, 5e-5]).reshape(1, 3))

    cartesian_quat_space_region.set_q_g(np.array([-0.17492908847362298, 0.6884405719242297, 0.6818253503208791, 0.17479727175084528]))  # set by jyp | grasping pose above the second object with marker
    cartesian_quat_space_region.set_Ko(60)

    image_space_region.set_x_d(np.array([960.5, 540.5]))
    image_space_region.set_Kv(np.array([[0.2, 0.1]]))

    writer = SummaryWriter(log_dir='./data/0708/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), flush_secs=5)
    
    # Step1: move the ArUco marker into FOV (with Cartesian region)

    while not rospy.is_shutdown():
        if data_c.is_data_without_vision_ready():
            J, r_t, r_o = data_c.J, data_c.trans, data_c.quat

            J_pinv = J.T @ np.linalg.pinv(J @ J.T)
            kesi_r = cartesian_space_region.kesi_r(r_t.reshape(1, 3))  # (1, 3)

            if cartesian_quat_space_region.fo(Quat(r_o)) <= 0:
                kesi_rq = np.zeros((1, 3))
            else:
                kesi_rq = cartesian_quat_space_region.kesi_rq_omega(r_o) / 2 # (1, 3)
            kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)

            if data_c.is_data_with_vision_ready():
                x = data_c.x
                # Js_hat.T version
                kesi_x = image_space_region.kesi_x(x).reshape(2, 1)
                Js_hat = get_Js_hat(data_c).astype(np.float)  # (2, 6)
                dq_d = - J_pinv @ (Js_hat.T @ kesi_x + kesi_rall.reshape(6, 1))

                """
                # Js_hat_pinv version
                # remember to times kesi_x by 1e-6

                Js_hat_pinv = Js_hat.T @ np.linalg.inv(Js_hat @ Js_hat.T)  # (6, 2)
                dq_d = - J_pinv @ (Js_hat_pinv @ kesi_x + kesi_rall.reshape(6, 1))
                print('Js_hat_pinv: ', Js_hat_pinv)
                print('kesi x: ', kesi_x.reshape(-1,))
                print('vision output: ', (J_pinv @ Js_hat_pinv @ kesi_x).reshape(-1,))
                """
            else:
                dq_d = - J_pinv @ kesi_rall

            if np.linalg.norm(data_c.x.reshape(-1,) - np.array([960.5, 540.5])) <= 1:
                dq_d = np.zeros((7,))
                msg = Float64MultiArray()
                msg.data = dq_d.reshape(7,)
                pub_.publish(msg)
                
                break
        
            msg = Float64MultiArray()
            msg.data = dq_d.reshape(7,)
            pub_.publish(msg)

            rate.sleep()

    # Step2: allow the ArUco marker center to have constant velocity in image space
    traj_g.state = 1
    cnt = 0
    while not rospy.is_shutdown():
        J = data_c.J
        J_pinv = J.T @ np.linalg.pinv(J @ J.T)

        x = data_c.x
        Js_hat = get_Js_hat(data_c)

        x_dot = traj_g.get_velocity(x).reshape(2, 1)
        dq_d = J_pinv @ (Js_hat.T @ x_dot)

        vel = vel_c.get_vel(x)  # list
        vel_x = {'desired': x_dot.reshape(-1,)[0],
                 'actual': vel[0]}
        vel_y = {'desired': x_dot.reshape(-1,)[1],
                 'actual': vel[1]}
        writer.add_scalars('vel_x', vel_x, global_step=cnt)
        writer.add_scalars('vel_y', vel_y, global_step=cnt)
        cnt += 1

        msg = Float64MultiArray()
        msg.data = dq_d.reshape(7,)
        pub_.publish(msg)

        rate.sleep()

    writer.close()

def plot_image_region_vector_field():
    """
        1. Plot vector field in the image plane.
        2. For each position x in the image plane, project kesi_x
           to Cartesian space, then project the intermediate
           result to joint space.
        3. Obtain jacobian from frankapy (python package), project
           result of 2 to Cartesian space, as if it is executed on
           Franka Emika.
        4. Finally, project the resulting Cartesian velocity to
           image plane, as the vector at x.
    """

    # hyper-parameters
    IMG_W = 1920
    IMG_H = 1080
    NUM_W = 48
    NUM_H = 27

    W_LIST = np.linspace(0, IMG_W - 1, NUM_W).reshape(-1,)
    H_LIST = np.linspace(0, IMG_H - 1, NUM_H).reshape(-1,)
    AXIS_W, AXIS_H = np.meshgrid(W_LIST, H_LIST)

    z = 0.5  # Supposing z is fixed

    R_b2c = np.array([[-1, 0,  0],
                      [0,  1,  0],
                      [0,  0, -1]])  # Rotational matrix from {world} to {camera}
    
    def compute_R_c2i(x:np.ndarray):
        x = x.reshape(-1,)
        u, v = x[0] - MyConstants.U0, x[1] - MyConstants.V0
        Z = MyConstants.CAM_HEIGHT - z
        R_c2i = np.array([[MyConstants.FX_HAT / Z, 0, - u / Z],
                          [0, MyConstants.FY_HAT / Z, - v / Z]])
        
        return R_c2i

    data_c = FrankaInfoStruct()
    data_c.trans[2] = z
    image_space_region = ImageSpaceRegion(b=np.array([IMG_W, IMG_H]))
    image_space_region.set_x_d(np.array([960.5, 540.5]))
    image_space_region.set_Kv(np.array([2, 1]))

    vector_field = np.zeros((NUM_H, NUM_W, 2))
    kesi_x_field = np.zeros((NUM_H, NUM_W, 2))
    for v_idx, v in enumerate(H_LIST):
        for u_idx, u in enumerate(W_LIST):
            # get kesi_x
            x = np.array([u, v]).reshape(1, 2)
            data_c.x = x
            kesi_x = image_space_region.kesi_x(x).reshape(2, 1)
            kesi_x_field[v_idx, u_idx, :] = - kesi_x.reshape(-1,)

            # get Js
            Js = get_Js_hat(data_c)
            pdb.set_trace()

            # get unit vector V_i(x)
            """
                dq_d = - J_pinv @ (Js.T @ kesi_x)
                 V_b = J @ dq_d
                 V_b = - (Js.T @ kesi_x)
                 V_c = R_b2c @ V_b
                 V_i = R_c2i @ V_c
            """
            R_c2i = compute_R_c2i(x)
            V_b = - (Js.T @ kesi_x).reshape(-1,)[:3]
            V_b = V_b.reshape(3, 1)
            V_c = (R_b2c @ V_b).reshape(3, 1)
            V_i = (R_c2i @ V_c).reshape(-1,)
            vector_field[v_idx, u_idx, :] = 1e3 * V_i

    # plot figure
    pdb.set_trace()
    plt.quiver(AXIS_W, AXIS_H, vector_field[..., 0], vector_field[..., 1], color='blue', pivot='mid', width=0.001)
    plt.quiver(AXIS_W, AXIS_H, kesi_x_field[..., 0], kesi_x_field[..., 1], color='red', pivot='mid', width=0.001)
    plt.show()


if __name__ == '__main__':
    # test_zero_Jacobian()
    # test_plot_3D()
    # test_Image_Jacobian()
    plot_image_region_vector_field()
