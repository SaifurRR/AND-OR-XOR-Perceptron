from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data=[[0,0],[0,1],[1,0],[1,1]] #AND i/p
labels=[0,1,1,1] #AND o/p


plt.scatter([point[0] for point in data],
            [point[1] for point in data],
            c=labels)
#AND (linearly separable): divide points of different color using a decision boundary.

#perceptron object:
classifier=Perceptron(max_iter=40, random_state=22)

# generate large data set to view decision boundary with heat-map
x_values=np.linspace(0,1,100)
y_values=np.linspace(0,1,100)

point_grid=list(product(x_values,y_values))

#train model:
classifier.fit(data, labels) 
#adjust weight: to get closer to labels for given i/p, linearly separate data using a decision boundary

# distance from decision boundary for the (x,y) points in point_grid
distances=classifier.decision_function(point_grid) 

#considering only absolute distance pt
abs_distances=[abs(a) for a in distances] #10,000 distances = 100*100

# 2dim for pcolormesh
# reshape (100,100) as every pt in 'x' is multiplied 100 times, so we have
# 100 different pairs for each variable in 'x' & 'y'
distance_matrix=np.reshape(abs_distances, (100,100))

# score: 0-1 as fraction of prediction
classifier.score(data, labels)
#typically we pass test data to evaluate performance, not the train data

# predict data (similar to labels)
classifier.predict(data)

#visualize perceptron's decision boundary:
classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]])

heatmap=plt.pcolormesh(x_values,y_values,distance_matrix)

plt.colorbar(heatmap)



plt.show()



