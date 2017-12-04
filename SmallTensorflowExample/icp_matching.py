import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time

def matching(true_data,generated_data):

    data_amount = np.shape(generated_data)[0]

    non_matched_index = np.arange(data_amount)
    matched_index = np.arange(data_amount)

    data_amount = np.shape(true_data)[0]
    shuffle_idx = np.arange(data_amount)
    np.random.shuffle(shuffle_idx)

    #calculate pairwise distances
    joint = np.concatenate((true_data, generated_data), axis=0)
    distances = pdist(joint)
    #make a square matrix from result
    distances_as_2d_matrix = squareform(distances)

    for true_idx in shuffle_idx:

        best_dist = float('inf')
        best_idx = -1


        for generated_idx in non_matched_index:

            if generated_idx == -1:
                continue

            dist = distances_as_2d_matrix[true_idx,generated_idx+data_amount]

            if dist < best_dist:
                best_dist = dist
                best_idx = generated_idx


        non_matched_index[best_idx] = -1
        matched_index[true_idx] = best_idx


    return matched_index


#For debugging
'''
data_amount = 100

a = np.random.rand(data_amount,2)
b = np.random.rand(data_amount,2)



start_time = time.time()
match_idx = matching(a,b)
print("--- %s seconds ---" % (time.time() - start_time))


for i in range(0,len(match_idx)):
    plt.scatter(a[i,0],a[i,1],c='blue')
    plt.scatter(b[match_idx[i],0],b[match_idx[i],1],c='orange')
    plt.plot([a[i,0],b[match_idx[i],0]],[a[i,1],b[match_idx[i],1]],c='black')

plt.show()
'''

