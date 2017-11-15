import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

root_folder = 'number_figures/'
files = ['predicted.csv']

for file in files:

    predicted = genfromtxt(root_folder+file, delimiter=',')


    images = predicted[:,]

    data_points = np.shape(predicted)[0]

    for point in range(0,data_points):
        image = images[point,:].reshape(28,28)

        plt.matshow(image)

        plt.ion()
        plt.show()

        time.sleep(1.0);
        plt.close('all')
        plt.ioff()






