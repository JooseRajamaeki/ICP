#This is a small sample code. The C++ implementation is much faster and better.

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from icp_matching import matching
import time
from enum import Enum

marker_size = 1

input_dimension = 6
output_dimension = 2
conditioning_dimension = 0

########### Making the neural network ###########

hidden_dimension = 100

learning_rate = 0.001
init_weight_scale = 0.1
max_seed_gradient_norm = 0.01


every_other = []
for i in range(hidden_dimension):
    if i % 2 == 0:
        every_other.append(1.0)
    else:
        every_other.append(-1.0)

comb  = tf.Variable(every_other, tf.float32)


x = tf.placeholder(tf.float32, [None, input_dimension])

W0 = tf.Variable(tf.random_normal([input_dimension, hidden_dimension])*init_weight_scale)
b0 = tf.Variable(tf.random_normal([hidden_dimension])*init_weight_scale)

hidden_activation_0 = tf.matmul(x, W0) + b0
hidden_output_0 = tf.nn.elu(hidden_activation_0)
bipolar_hiden_output_0 = hidden_output_0*comb


W1 = tf.Variable(tf.random_normal([hidden_dimension, hidden_dimension])*init_weight_scale)
b1 = tf.Variable(tf.random_normal([hidden_dimension])*init_weight_scale)

hidden_activation_1 = tf.matmul(bipolar_hiden_output_0, W1) + b1
hidden_output_1 = tf.nn.elu(hidden_activation_1)
bipolar_hiden_output_1 = hidden_output_1*comb



W_output = tf.Variable(tf.random_normal([hidden_dimension, output_dimension])*init_weight_scale)
b_output = tf.Variable(tf.random_normal([output_dimension])*init_weight_scale)


y = tf.matmul(bipolar_hiden_output_1, W_output) + b_output
y_ = tf.placeholder(tf.float32, [None, output_dimension])

loss = tf.reduce_sum(tf.square(y-y_))




train_step_tensorflow = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


########### Initializing the neural network ###########


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

########### Plotting sample input noise ###########

amount_data = 1000
x_data = np.random.randn(amount_data,input_dimension)

plt.scatter(x_data[:,0], x_data[:,1],s=marker_size)

plt.title('Input noise')

plt.ion()
plt.show()
time.sleep(2.0);
plt.close('all')
plt.ioff()

########### Creating training data ###########

y_data = np.random.rand(amount_data,output_dimension)


y_data[0:round(amount_data/2),0] = y_data[0:round(amount_data/2),0] + 2
y_data[0:round(amount_data/2),1] = y_data[0:round(amount_data/2),1] - 1

'''
y_data[:,0] = y_data[:,0]*7.0
corruption_noise = np.random.randn(amount_data,1) * 0.1
y_data[:,1] = np.sin(y_data[:,0])+corruption_noise[:,0]
'''


########### Defining the training step of the algorithm ###########

def distribution_training_step(true_data,input_noise):

    if conditioning_dimension > 0:
        input_noise[:,0:conditioning_dimension] = true_data[:,0:conditioning_dimension]

    generated_data = sess.run(y, feed_dict={x: input_noise})

    if conditioning_dimension > 0:
        generated_data[:,0:conditioning_dimension] = true_data[:,0:conditioning_dimension]

    matched_indexes = matching(true_data,generated_data)

    input_noise = input_noise[matched_indexes,:]

    if conditioning_dimension > 0:
        input_noise[:,0:conditioning_dimension] = true_data[:,0:conditioning_dimension]


    minibatch_size = round(amount_data / 10)
    num_minibatches = round(amount_data/minibatch_size)

    if num_minibatches * minibatch_size < amount_data:
        num_minibatches = num_minibatches + 1

    for batch_idx in range(num_minibatches):
        index = np.arange(amount_data)
        np.random.shuffle(index)
        index = index[0:minibatch_size]

        train_noise = input_noise[index,:]
        train_data = true_data[index,:]

        sess.run(train_step_tensorflow, feed_dict={x: train_noise, y_: train_data})


########### Actual training ###########


for iteration in range(10000):
    x_data = np.random.randn(amount_data,input_dimension)
    distribution_training_step(y_data,x_data)
    print(iteration)
    if iteration % 100 == 0:
        x_data = np.random.randn(amount_data,input_dimension)

        if conditioning_dimension > 0:
            x_data[:,0:conditioning_dimension] = y_data[:,0:conditioning_dimension]
        
        predictions = sess.run(y, feed_dict={x: x_data})

        if conditioning_dimension > 0:
            predictions[:,0:conditioning_dimension] = y_data[:,0:conditioning_dimension]
        
        true_data = plt.scatter(y_data[:,0], y_data[:,1],s=marker_size)
        generated_data = plt.scatter(predictions[:,0], predictions[:,1],s=marker_size)
        plt.legend([true_data, generated_data], ['True data', 'Generated data'])

        plt.savefig('distributions.png', bbox_inches='tight', pad_inches=0)
        
        plt.ion()
        plt.show()

        time.sleep(2.0);
        plt.close('all')
        plt.ioff()
      
  
