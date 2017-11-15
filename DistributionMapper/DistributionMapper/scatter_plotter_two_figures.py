import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

true_data = genfromtxt('true_data.csv', delimiter=',')
initial = genfromtxt('initial.csv', delimiter=',')
early = genfromtxt('early.csv', delimiter=',')
predicted = genfromtxt('predicted.csv', delimiter=',')

marker_size = 1
    

f, axarr = plt.subplots(4, sharex=True)
axarr[0].scatter(true_data[:,0],true_data[:,1],s=marker_size)
axarr[0].set_title('True data')

axarr[1].scatter(predicted[:,0],predicted[:,1],c='orange',s=marker_size)
axarr[1].set_title('Generated data')

axarr[2].scatter(early[:,0],early[:,1],c='orange',s=marker_size)
axarr[2].set_title('Generated data, epoch 5')

axarr[3].scatter(initial[:,0],initial[:,1],c='orange',s=marker_size)
axarr[3].set_title('Generated data, before training')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig('distribution_two.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('distribution_two.png', bbox_inches='tight', pad_inches=0)

plt.close('all')






