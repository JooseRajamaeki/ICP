import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

marker_size = 1

predicted = genfromtxt('predicted.csv', delimiter=',')
true_data = genfromtxt('true_data.csv', delimiter=',')

plt.scatter(true_data[:,0],true_data[:,1],c='blue',s=marker_size)
plt.scatter(predicted[:,0],predicted[:,1],c='orange',s=marker_size)

plt.savefig('distributions.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('distributions.png', bbox_inches='tight', pad_inches=0)

plt.ion()
plt.show()

time.sleep(2.0);
plt.close('all')
plt.ioff()





