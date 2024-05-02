import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
import time

move_data = np.loadtxt('log.txt', dtype=float, delimiter=',')

# PLOT
point_fig = plt

point_fig.scatter(move_data[:,0], move_data[:,1], c=move_data[:,2], marker=1, s=(move_data[:,3]+2)**4)


plt.show()