import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

marker_size = 1

convergence = genfromtxt('continuous_to_discrete_convergence.csv', delimiter=',')
empirical_errors = genfromtxt('empirical_errors.csv', delimiter=',')

empirical_error = np.mean(empirical_errors)

curves = np.shape(convergence)[0]
print(curves)

for i in range(0,curves):
    plt.plot(convergence[i,:])

plt.plot([0,np.shape(convergence)[1]],[empirical_error,empirical_error],'r')

plt.xlabel('Epoch')
plt.ylabel('Error')

plt.savefig('continuous_to_discrete_convergence.eps', bbox_inches='tight', pad_inches=0)
#plt.savefig('continuous_to_discrete_convergence.png', bbox_inches='tight', pad_inches=0)

plt.ion()
plt.show()

time.sleep(2.0);
plt.close('all')
plt.ioff()





