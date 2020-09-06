
import math
import numpy as np
#import ast
#with open('examples.npy') as e:
#	X = ast.literal_eval(e.read())
import list_imports
X = list_imports.get("examples.npy")

#pseudocode 
#Data inputs are (quality, angle, distance)

#Xr = np.roll(X,1)         
#diff = Xr - X
#change_point = diff < 300 #

#if diff > 360

#dist = X[1] #index 2nd element of tuple

#x_coord = dist*cos(angle)
#y_coord = dist*sin(angle)

#code begins


X = np.vstack((x[0],x[1],x[2]))
th = X[:,1] #all rows, column 1

# th = np.array([0,1,3,4,354,357, 358, 259, 360, 0, 1, 2,3,4,354,357, 358, 259, 360, 0, 1, 2])
print(th)

th_r = np.roll(th,1) #shift last element to first
print(th_r)

diff = th_r - th 
print(diff)

change_point = diff > 350 #change point if diff greater than 350
print(change_point)

change_point_loc = np.where(change_point == True)[0]
print(change_point_loc)

scan1 = X[0:change_point_loc[0]]
print(scan1)

scan2 = X[change_point_loc[0]:change_point_loc[1]]
print(scan2)

x_coord = X[:,2]*cos(X[:,1])
print('x_coord is',x_coord)

y_coord = X[:,2]*sin(X[:,1])
print('y_coord is',y_coord)
