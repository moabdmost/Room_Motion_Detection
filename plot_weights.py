import numpy as np
import matplotlib.pyplot as plt
import datetime


ts2dt = lambda timestamp: datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

tt = np.vectorize(ts2dt)

data_load = np.loadtxt('weights_example.txt', dtype = float, delimiter=',')

diff_list = []

for i in range(len(data_load[:, 0])):
    diff = data_load[i,0] - data_load[i-1,0]
    diff_list.append(diff)

print (1/np.median(diff_list)) # sample rate.

plt.plot(data_load[:,0], data_load[:,1]) # plot with unixtime

# time = tt(data_load[:,0]) # convert unixtime to datetime
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(time, data_load[:,1])


# print (tt(data_load[:,0]))

# plt.plot(time, data_load[:,1])
# fig.autofmt_xdate()

plt.show()
