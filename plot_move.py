import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
import time

move_data = np.loadtxt('log.txt', dtype=float, delimiter=',')

# PLOT
point_fig = plt
# point_fig = fig1.add_subplot(111)
point_fig.scatter(move_data[:,0], move_data[:,1], c=move_data[:,2])
# point_fig.colorbar(label=np.unique(label_data['f1']))

# fig2 = plt.figure()
# hist_fig = fig2.add_subplot(111)
# hist_fig.plot(label_data['f0'][-1]-label_data['f0'], metric_data[:,1], ".")

plt.show()