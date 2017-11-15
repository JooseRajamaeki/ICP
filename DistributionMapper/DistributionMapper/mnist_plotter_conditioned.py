import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time
import string
import random
import os
from time import gmtime, strftime

root_folder = 'number_figures/'
files = ['zeros','ones','twos','threes','fours','fives','sixes','sevens','eights','nines']
time_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

for file in files:

    predicted = genfromtxt(root_folder+file+'.csv', delimiter=',')


    images = predicted[:,1:]
    labels = predicted[:,0]


    data_points = np.shape(predicted)[0]

    for point in range(0,data_points):
        image = images[point,:].reshape(28,28)

        plt.matshow(image)

        directory = root_folder+file

        try:
            os.stat(directory)
        except:
            os.mkdir(directory)   

        random_string = directory+'/'
        random_string = random_string + time_string
        random_string = random_string + '/'

        try:
            os.stat(random_string)
        except:
            os.mkdir(random_string)  
        
        #for _ in range(0,10):
        #    random_string = random_string + random.choice(string.ascii_lowercase)

        random_string = random_string + str(point)

        plt.axis('off')

        plt.savefig(random_string+'.eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(random_string+'.png', bbox_inches='tight', pad_inches=0)

        plt.close('all')

        #plt.ion()
        #plt.show()

        #time.sleep(1.0);
        #plt.close('all')
        #plt.ioff()






