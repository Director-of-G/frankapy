import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import pdb

def test_Jacobian():
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


if __name__ == '__main__':
    # test_Jacobian()
    test_plot_3D()
