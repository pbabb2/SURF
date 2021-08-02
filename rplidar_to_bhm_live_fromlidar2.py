import os
import time
import numpy as np
import torch as pt
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from bhmtorch_cpu import BHM2D_PYTORCH
from rplidar import RPLidar
import sys
dtype = pt.float32
device = pt.device("cpu")
PORT_NAME = '/dev/ttyUSB0'

def read_lidar(num_of_scans_to_stack, itera=0):
    # Read from lidar
    lidar_data = [] #initialize array for live lidar
    itera=0 #set iterator to zero
    lidar = RPLidar(PORT_NAME) #define variable to collect data from lidar
    rp = lidar
    for i, scan in enumerate(lidar.iter_scans()): #enumerate lidar scans to make them iterable
        print('%d: Got %d measurments' % (i, len(scan)))
        lidar_data.append(np.array(scan))
        if i > 10: #if greater than 10 lidar scans, stop enumerating scans
            lidar.stop()
            break
    rp_lidar_data = lidar_data #assign lidar_data to rp_lidar_data
    
    # Read data from rplidar file
    #filename = 'examples'
    #print(" |_reading LIDAR from " + 'datasets/' + filename + '.npy')
    #rp_lidar_data = np.load('datasets/' + filename + '.npy', allow_pickle=True)
    

    # Virtically stack all data
    start = itera*num_of_scans_to_stack
    rp = rp_lidar_data[start]
    num_of_scans_to_stack = min(num_of_scans_to_stack, len(rp))
    for i in range(start+1, start+num_of_scans_to_stack):
        rp = np.vstack((rp, rp_lidar_data[i]))

    # Scale: mm->m
    rp[:, 2] /= 1000
    th = rp[:, 1]

    # Detect change point
    th_r = np.roll(th, 1)  # shift last element to first
    diff = th_r - th
    change_point = diff > 250  # change point if diff greater than 250
    change_point_loc = np.where(change_point == True)[0]
    change_point_loc = np.insert(change_point_loc, 0, [0, ])
    print("change_point_loc: ", change_point_loc)

    # for each lidar circle
    for i in range(len(change_point_loc) - 1):
        scani = rp[change_point_loc[i]:change_point_loc[i + 1], :]
        print('   |_subscan {}: {}'.format(i, scani.shape))

        # todo: add robot pose to the following coords
        x_coord = scani[:, 2] * np.cos(np.deg2rad(scani[:, 1]))
        y_coord = scani[:, 2] * np.sin(np.deg2rad(scani[:, 1]))
        y_coord = y_coord*-1 #mirror y coordinates about the x-axis to mirror the figure
        X = np.hstack((x_coord[:, None], y_coord[:, None]))  # Nx2 numpy array of lidar positions
        y = np.ones(X.shape[0])  # N numpy array (1 indicate when the lidar hits and 0 indicate when the lidar does not reflect back)
        data = np.hstack((X,y[:,None]))

        if i == 0:
            data_all = data
        else:
            data_all = np.vstack((data_all, data))

    print(" |_{} lidar hits in this frame".format(data_all.shape))
    return data_all

def get_filled_txy(time, X, y, car_pos=np.array([0, 0]), max_range=20, unoccupied_points_per_meter=5, margin=0.05):
    """
    Add free points along lidar beams
    """
    print(" |_in-filling started")
    distances = np.sqrt(np.sum((X - car_pos) ** 2, axis=1))
    for n in range(len(distances)):
        dist = distances[n]
        laser_endpoint = X[n, :]
        para = np.sort(
            (np.random.random(np.int16(dist * unoccupied_points_per_meter)) * (1 - 2 * margin) + margin)[:, np.newaxis])
        points_scan_i = car_pos + para * (laser_endpoint - car_pos)
        if n == 0:  # first data point
            # points = np.vstack((points_scan_i, laser_endpoint))
            if y[n] == 0:  # laser beam didn't return
                points = points_scan_i
                labels = np.zeros((points_scan_i.shape[0], 1))
            else:

                points = np.vstack((points_scan_i, laser_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), y[n] * np.array([1])[:, np.newaxis]))
        else:
            # points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
            if y[n] == 0:  # laser beam didn't return
                points = np.vstack((points, points_scan_i))
                labels = np.vstack((labels, np.zeros((points_scan_i.shape[0], 1))))
            else:
                points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
                mini_labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), y[n] * np.array([1])[:, np.newaxis]))
                labels = np.vstack((labels, mini_labels))

    data = np.hstack((points, labels))
    data = np.hstack((time + np.zeros(data.shape[0])[:, None], data))
    print(' |_after in-filling {}'.format(data.shape))
    return data

def bhm_train(df, config_name, predict=True):
    print(" |_training started")

    # Read parameters
    with open(config_name) as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        fn_train, fn_out, hinge_resolution, gamma, q_resolution, t_steps = yaml_data['fn_train'], yaml_data['fn_out'], \
                                                                           yaml_data['hinge_resolution'], yaml_data[
                                                                               'gamma'], yaml_data['q_resolution'], \
                                                                           yaml_data['t_steps']
        fn_train = os.getcwd() + fn_train
        fn_out = os.getcwd() + fn_out

    df = df[:, 1:]

    # Determine the lidar area
    delta = 1
    area_min, area_max = df[:, :2].min(axis=0), df[:, :2].max(axis=0)
    area_max_min = [area_min[0] - delta, area_max[0] + delta, area_min[1] - delta, area_max[1] + delta]
    print('   |_cell area = {}'.format(area_max_min))

    # Prepare data
    df = pt.tensor(df, dtype=pt.float32)
    X = df[:, :2]
    y = df[:, 2].reshape(-1, 1)
    print('   |_data shape={}'.format(X.shape))

    # Define the model
    bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=None, cell_resolution=hinge_resolution, cell_max_min=area_max_min,
                            X=X, nIter=1)

    # Fit the model
    t1 = time.time()
    bhm_mdl.fit(X, y)
    t2 = time.time()
    trainTime = np.round(t2 - t1, 3)
    print('   |_training time={} s'.format(trainTime))

    if predict is True:
        # Query the model
        xx, yy = np.meshgrid(np.arange(area_max_min[0], area_max_min[1] + q_resolution, q_resolution),
                             np.arange(area_max_min[2], area_max_min[3] + q_resolution, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        Xq = pt.tensor(grid, dtype=pt.float32)
        t1 = time.time()
        yq = bhm_mdl.predict(Xq)
        t2 = time.time()
        queryTime = np.round(t2 - t1, 3)
        print('   |_query time={} s'.format(queryTime))
        
        
    
        return Xq, yq
    else:
        return None, None

def update_myplot(i, num_of_scans_to_stack, config_name, ax1_plot, ax2_plot):
    print("\n ----------------------------")
    print("Frame {}".format(i))

    # read lidar
    data = read_lidar(num_of_scans_to_stack = num_of_scans_to_stack, itera = i) # (X, y, occ=1)

    # infill lidar
    data = get_filled_txy(i, data[:,:2], data[:,2]) # (time, X, y, occ={0,1})

    # train and query
    Xq, yq = bhm_train(data, config_name=config_name, predict=True)

    # plot
    #consider mirroring image about the x-axis, https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    
    ax1_plot.set_offsets(data[:, 1:3])
    ax1_plot.set_color(cm.jet(data[:, 3]))
    ax2_plot.set_offsets(Xq)
    ax2_plot.set_color(cm.jet(yq))

def main():
    # parameters
    num_of_scans_to_stack = 5
    config_name = 'config/real_lidar_cut.yaml'
    n_time_steps = 2000
    delay_between_frames_ms = 500
    plot_minmax = (-2,2)

    # plot
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax1.set_xlim(plot_minmax); ax1.set_ylim(plot_minmax)
    ax2 = plt.subplot(122)
    ax2.set_xlim(plot_minmax) ;ax2.set_ylim(plot_minmax)
    ax1_plot = ax1.scatter([0, 0], [0, 0], s=50, c=[0, 1], cmap=plt.cm.Greys_r, lw=0)
    ax2_plot = ax2.scatter([0, 0], [0, 0], s=50, c=[0, 1], cmap=plt.cm.Greys_r, lw=0)

    # animate the function
    _ = FuncAnimation(fig=fig, func=update_myplot,  frames=np.arange(0, n_time_steps, 1), interval=delay_between_frames_ms,\
                  repeat=False, fargs=(num_of_scans_to_stack, config_name, ax1_plot, ax2_plot))
    plt.show()

if __name__ == '__main__':
    main()