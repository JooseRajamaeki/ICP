import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time


files = ['images.csv']

for file in files:

    predicted = genfromtxt(file, delimiter=',')

    images = predicted[:,]

    data_points = np.shape(predicted)[0]

    for point in range(0,data_points):
        image = images[point,:].reshape(28,28)

        plt.matshow(image)

        plt.axis('off')

        random_string = str(point)

        #plt.savefig(random_string+'.eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(random_string+'.png', bbox_inches='tight', pad_inches=0)


        plt.close('all')

