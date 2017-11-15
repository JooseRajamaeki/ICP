import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time

inputs = genfromtxt('noise_inputs.csv', delimiter=',')
predicted = genfromtxt('predicted.csv', delimiter=',')

marker_size = 1

data_amount = np.shape(inputs)[0]

for i in range(0,data_amount):
    color = 'b'
    if (inputs[i,0] < 0.0):
        color = 'r'
    plt.scatter(inputs[i,0],inputs[i,1],c=color,s=marker_size)
    
plt.axis('off')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig('swiss_roll_in.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('swiss_roll_in.png', bbox_inches='tight', pad_inches=0)


plt.ion()
plt.show()

time.sleep(2.0);
plt.close('all')
plt.ioff()



for i in range(0,data_amount):
    color = 'b'
    if (inputs[i,0] < 0.0):
        color = 'r'
    plt.scatter(predicted[i,0],predicted[i,1],c=color,s=marker_size)

plt.axis('off')



plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig('swiss_roll_out.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('swiss_roll_out.png', bbox_inches='tight', pad_inches=0)


plt.ion()
plt.show()

time.sleep(2.0);
plt.close('all')
plt.ioff()






