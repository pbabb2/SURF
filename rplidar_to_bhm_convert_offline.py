# RPlidar to Bayesian Hilber Maps preprocessing
# This assumes a stationary or slowly moving lidar

import numpy as np
import matplotlib.pyplot as pl

def get_filled_txy(X, y, car_pos=np.array([0,0]), max_range=20, unoccupied_points_per_meter=0.5, margin=0.05):
    """
    Add free points along lidar beams
    """
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

def main(filename, num_of_scans_to_stack):
    # Read data from rplidar file
    rp_lidar_data = np.load(filename + '.npy', allow_pickle=True)

    # Virtically stack all data
    rp = rp_lidar_data[0]
    num_of_scans_to_stack = min(num_of_scans_to_stack, len(rp))
    for i in range(1,num_of_scans_to_stack):
        rp = np.vstack((rp, rp_lidar_data[i]))

    # Scale: mm->m
    rp[:,2] /= 100
    th = rp[:,1]

    # Detect change point
    th_r = np.roll(th,1) #shift last element to first
    diff = th_r - th
    change_point = diff > 350 #change point if diff greater than 350
    change_point_loc = np.where(change_point == True)[0]
    change_point_loc = np.insert(change_point_loc, 0, [0,])
    print("change_point_loc: ", change_point_loc)

    # for each lidar circle
    for i in range(len(change_point_loc)-1):
        scani = rp[change_point_loc[i]:change_point_loc[i+1],:]
        print('scan {}: {} before in-filling'.format(i, scani.shape))

        #todo: add robot pose to the following coords
        x_coord = scani[:,2]*np.cos(np.deg2rad(scani[:,1]))
        y_coord = scani[:,2]*np.sin(np.deg2rad(scani[:,1]))
        X = np.hstack((x_coord[:,None], y_coord[:,None])) # Nx2 numpy array of lidar positions
        y = np.ones(X.shape[0]) # N numpy array (1 indicate when the lidar hits and 0 indicate when the lidar does not reflect back)
        data = get_filled_txy(X, y)
        data = np.hstack((i+np.zeros(data.shape[0])[:,None], data))
        print('scan {}: {} after in-filling'.format(i, data.shape))

        # plot
        fig = pl.figure(figsize=(10,4))
        pl.suptitle('scan {}'.format(i))
        ax = fig.add_subplot(121, projection='polar')
        pl.polar(np.deg2rad(scani[:,1]), scani[:,2], c='r')
        pl.subplot(122)
        pl.scatter(data[:,1], data[:,2], c=data[:,3], s=10, cmap='jet')
        #pl.show()
        pl.savefig('datasets/figs/'+ filename+'bhm_ready_{}.png'.format(i))

        if i == 0:
            data_all = data
        else:
            data_all = np.vstack((data_all, data))

        np.savetxt('datasets/' + filename+'_bhm_ready.csv', data, delimiter=',')
        print('\n')

if __name__ == '__main__':
    filename = 'examples'
    num_of_scans_to_stack = 10
    main(filename, num_of_scans_to_stack)