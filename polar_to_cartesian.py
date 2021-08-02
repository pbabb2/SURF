
import math
import numpy as np
#import ast
#with open('examples.npy') as e:
#	X = ast.literal_eval(e.read())
import list_imports
#X = list_imports.get("examples.npy")
x = np.load("examples.npy");

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

#print(th)

th_r = np.roll(th,1) #shift last element to first
#print(th_r)

diff = th_r - th 
#print(diff)

change_point = diff > 350 #change point if diff greater than 350
#print(change_point)

change_point_loc = np.where(change_point == True)[0]
#print(change_point_loc)

scan1 = X[0:change_point_loc[0]]
#print(scan1)

scan2 = X[change_point_loc[0]:change_point_loc[1]]
#print(scan2)

x_coord = X[:,2]*np.cos(X[:,1])
print('x_coord is',x_coord)

y_coord = X[:,2]*np.sin(X[:,1])
print('y_coord is',y_coord)


def get_filled_txy(X, y, car_pos=np.array([0,0]), max_range=20, unoccupied_points_per_meter=0.1, margin=0.05):
    distances = np.sqrt(np.sum( (X - car_pos)**2, axis=1))
    for n in range(len(distances)):
        dist = distances[n]
        laser_endpoint = X[n, :]
        para = np.sort((np.random.random(np.int16(dist * unoccupied_points_per_meter)) * (1 - 2 * margin) + margin)[:,np.newaxis])
        points_scan_i = car_pos + para * (laser_endpoint - car_pos)
        if n == 0:  # first data point
            # points = np.vstack((points_scan_i, laser_endpoint))
            if y[n] == 0: # laser beam didn't return
                points = points_scan_i
                labels = np.zeros((points_scan_i.shape[0], 1))
            else:
                points = np.vstack((points_scan_i, laser_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), y[n] * np.array([1])[:, np.newaxis]))
        else:
            # points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
            if y[n] == 0: # laser beam didn't return
                points = np.vstack((points, points_scan_i))
                labels = np.vstack((labels, np.zeros((points_scan_i.shape[0], 1))))
            else:
                points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
                mini_labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), y[n]*np.array([1])[:, np.newaxis]))
                labels = np.vstack((labels,  mini_labels))
    # return (N,3) x1, x2, occ
    return np.hstack((points, labels))

X = np.hstack((x_coord[:,None], y_coord[:,None])) # Nx2 numpy array of lidar positions
y = np.ones(X.shape[0]) # N numpy array (1 indicate when the lidar hits and 0 indicate when the lidar does not reflect back)
data = get_filled_txy(X, y)
data = np.hstack((np.zeros(data.shape[0])[:,None], data))
np.savetxt('real_lidar.csv', data, delimiter=',')