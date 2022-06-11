import matplotlib.pyplot as plt
import numpy as np

traj_path = '/home/roboticslab/yxj/frankapy/data/0608/q_and_m.npy'  
q_and_m = np.load(traj_path)

index = np.argmin(q_and_m[:, 7])
print(index)
print(np.min(q_and_m[:,7]))
print(q_and_m[index,:7])
print(q_and_m[8000,:7])

plt.figure()
plt.plot(q_and_m[:, 7])
plt.show()

