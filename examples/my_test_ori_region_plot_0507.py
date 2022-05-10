from cProfile import label
from msilib.schema import Font
import numpy as np
from pyquaternion import Quaternion
from transformations import quaternion_from_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawArrow(A, B):
    fig = plt.figure(figsize=(5, 5))
    print("xasxcsasdc")
    ax = fig.add_subplot(121)
    # fc: filling color
    # ec: edge color


    """第一种方式"""
    ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1],
             width=0.01,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.25,
              head_length=0.25,
             fc='b',
             ec='b')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.grid()
    ax.set_aspect('equal')

    """第二种方式"""
    # 这种方式是在图上做标注时产生的
    # Example:
    ax = fig.add_subplot(122)
    ax.annotate("",
                xy=(B[0], B[1]),
                xytext=(A[0], A[1]),
                # xycoords="figure points",
                arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.grid()
    ax.set_aspect('equal') #x轴y轴等比例

    #x轴y轴等比例
    plt.show()

def drawCoordinateGoal(R, ax):
    ax.quiver(0,0,0,R[0,0],R[1,0],R[2,0],arrow_length_ratio=0.1,normalize=True,color = 'r',label='x')
    ax.quiver(0,0,0,R[0,1],R[1,1],R[2,1],arrow_length_ratio=0.1,normalize=True,color = 'g',label='y')
    ax.quiver(0,0,0,R[0,2],R[1,2],R[2,2],arrow_length_ratio=0.1,normalize=True,color = 'b',label='z')

def drawCoordinate(R, ax):
    ax.quiver(0,0,0,R[0,0],R[1,0],R[2,0],arrow_length_ratio=0.1,normalize=True,color = 'r')
    ax.quiver(0,0,0,R[0,1],R[1,1],R[2,1],arrow_length_ratio=0.1,normalize=True,color = 'g')
    ax.quiver(0,0,0,R[0,2],R[1,2],R[2,2],arrow_length_ratio=0.1,normalize=True,color = 'b')
    # return ax

def visualize(q_list):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(len(q_list)):
        R = q_list[i].rotation_matrix
        if i==0:
            drawCoordinateGoal(R,ax)
        else:
            drawCoordinate(R,ax)

    limit = 1
    ax.legend(loc=1)
    ax.set_xlim([-limit,limit])
    ax.set_ylim([-limit,limit])
    ax.set_zlim([-limit,limit])
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_zlabel('z/m')
    plt.title('coordinates',fontdict={'weight':'normal','size': 20})
    plt.savefig('OrtRegion.pdf',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.show()

if __name__ =="__main__":
    q_g = quaternion_from_matrix( np.array( [[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]] ) )
    print('q_g: ',q_g)

    # q = Quaternion.random()
    # print('q: ',q)
    # print('exp(q):',Quaternion.exp(q))

    # print('log(q):',Quaternion.log(q))
    # print("distance: ",Quaternion.distance(q_g,q))

    q_list = [Quaternion(q_g)]
    # for i in range(5000):
    #     q = Quaternion.random()
    #     # print('q: ',q)
    #     error = q_g * q.inverse
    #     # print("(q_g * q.inverse): ", error[2])
    #     # print("Quaternion.log(q_g * q.inverse): ", Quaternion.log(error[2]))
    #     a =  Quaternion.log(error[2])
    #     if (a[1]*a[1]+a[2]*a[2]+a[3]*a[3])<0.2:
    #         q_list.append(q)

    distance1 = Quaternion.absolute_distance(Quaternion(q_g),Quaternion(q_g))
    print("absolute_distance",distance1)
    distance2 = Quaternion.distance(Quaternion(q_g),Quaternion(q_g))
    print("distance",distance2)

    for i in range(30000):
        q = Quaternion.random()
        # print('q: ',q)
        distance1 = Quaternion.absolute_distance(q,q_g)
        # print("distance1: ", distance1)
        # print("Quaternion.log(q_g * q.inverse): ", Quaternion.log(error[2]))
        # a =  Quaternion.log(error[2])
        if distance1<0.2:
            q_list.append(q)

    print(len(q_list))
    visualize(q_list)



