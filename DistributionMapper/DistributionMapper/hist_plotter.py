import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

predicted = genfromtxt('predicted_hist.csv', delimiter=',')
empirical = genfromtxt('empirical_hist.csv', delimiter=',')
true_data = genfromtxt('true_hist.csv', delimiter=',')

ind = np.arange(len(predicted))
width = 0.15

fig, ax = plt.subplots()
rects1 = ax.bar(ind, predicted, width, color='r')
rects2 = ax.bar(ind + width, empirical,width, color='y')
rects3 = ax.bar(ind + 2.0*width, true_data,width, color='b')

ax.legend((rects1[0], rects2[0], rects3[0]), ('Generated', 'Empirical','True'))

plt.savefig('continuous_to_discrete_histogram.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('continuous_to_discrete_histogram.png', bbox_inches='tight', pad_inches=0)

error = 0.0

for i in range(0,len(predicted)):
    error = error + abs(predicted[i]-true_data[i])

print(error)

plt.ion()
plt.show()

time.sleep(2.0);
plt.close('all')
plt.ioff()







