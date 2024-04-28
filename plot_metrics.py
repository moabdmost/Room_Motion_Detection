import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time

metric_data = np.loadtxt('camerasheet.txt', dtype=float, delimiter=',')
label_data = np.genfromtxt('labelsheet.txt', delimiter=',', dtype=None, encoding=None)
# label_data = np.genfromtxt('labelsheet.txt', delimiter=',', deletechars=" a-zA-Z", filling_values=10)

# timeunix = np.array([pair[0] for pair in label_data])

# Find common unixtime values
common_unixtime = np.intersect1d(label_data['f0'], metric_data[:,0])

# Filter label_data and metric_data to keep only the rows with common unixtime values
label_data = label_data[np.isin(label_data['f0'], common_unixtime)]
metric_data = metric_data[np.isin(metric_data[:,0], common_unixtime)]

# print(label_data['f1'])
# timeunix = np.array([pair[0] for pair in label_data])

# Encode categorical labels into numerical for the scatter plot's colormap.
label_encoder = LabelEncoder()
numeric_colors = label_encoder.fit_transform(label_data['f1'])

# print (label_data[-1, 0] - label_data[:,0])
# print ((label_data[0])[:])
# print (metric_data[-1,0])

# PLOT
point_fig = plt
# point_fig = fig1.add_subplot(111)
point_fig.scatter(label_data['f0'][-1]-label_data['f0'], metric_data[:,1], c=numeric_colors, cmap='viridis')
point_fig.colorbar(label=np.unique(label_data['f1']))

fig2 = plt.figure()
hist_fig = fig2.add_subplot(111)
hist_fig.plot(label_data['f0'][-1]-label_data['f0'], metric_data[:,1], ".")

plt.show()