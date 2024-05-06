import numpy as np
import matplotlib.pyplot as plt

move_data = np.loadtxt('log.txt', dtype=float, delimiter=',') #Read data from log file

move_data = move_data[(move_data[:,2] != 0) & (move_data[:,3] != 2)] #FILTIRING: Index 2 is the state, index 3 is the direction.

#state is 0 for initial state, 1 for 1st side, 2 for 2nd side, 3 for 3rd side. 
#direction is 0 for initial state, 1 for in, 2 for out


# PLOT
point_fig = plt
point_fig.scatter(move_data[:,0], move_data[:,1], c=move_data[:,2], marker=1, s=(move_data[:,3]+2)**4)

plt.show()