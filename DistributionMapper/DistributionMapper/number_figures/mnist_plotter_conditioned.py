import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time
import string
import numpy as np
import random
import os
from time import gmtime, strftime


files = ['zeros','ones','twos','threes','fours','fives','sixes','sevens','eights','nines']
time_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

f, axarr = plt.subplots(10,10,sharex=True, sharey=True)

idx_x = 0


for file in files:

    predicted = genfromtxt(file+'.csv', delimiter=',')


    images = predicted[:,1:]
    labels = predicted[:,0]


    data_points = np.shape(predicted)[0]

    idx_y = 0

    for point in range(0,data_points):
        image = images[point,:].reshape(28,28)

        axarr[idx_x,idx_y].matshow(image)
        axarr[idx_x,idx_y].axis('off')
        #axarr[idx_x,idx_y].remove()

        idx_y = idx_y + 1

    idx_x = idx_x + 1

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

plt.savefig('mnist.eps', bbox_inches='tight', pad_inches=0)
#plt.savefig(random_string+'.png', bbox_inches='tight', pad_inches=0)

#plt.tight_layout()

plt.show()
plt.close('all')

        #plt.ion()
        #plt.show()

        #time.sleep(1.0);
        #plt.close('all')
        #plt.ioff()






