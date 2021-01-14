# RPlidar to Bayesian Hilber Maps preprocessing
# This assumes a stationary or slowly moving lidar
#import tkinter
import numpy as np
#import matplotlib.pyplot as pl

import sys
import os
import time
import numpy as np
import pandas as pd
import torch as pt

import yaml
from bhmtorch_cpu import BHM2D_PYTORCH


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import time
from array import array as arr
import matplotlib.cm as cm

DMAX = 4000
IMIN = 0
IMAX = 50


def get_filled_txy(X, y, car_pos=np.array([0,0]), max_range=20, unoccupied_points_per_meter=5, margin=0.05):
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
    rp_lidar_data = np.load('datasets/' + filename + '.npy', allow_pickle=True)

    # Virtically stack all data
    rp = rp_lidar_data[0]
    num_of_scans_to_stack = min(num_of_scans_to_stack, len(rp))
    for i in range(1,num_of_scans_to_stack):
        rp = np.vstack((rp, rp_lidar_data[i]))

    # Scale: mm->m
    rp[:,2] /= 1000
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
        pl.savefig('datasets/figs/'+ filename+'_bhm_ready_{}.png'.format(i))
        
        
        #update chart
        
        #run chart

        if i == 0:
            data_all = data
        else:
            data_all = np.vstack((data_all, data))

        np.savetxt('datasets/' + filename+'_bhm_ready.csv', data_all, delimiter=',')
        print('\n')
        
        ###################################################################################
        #BHMs
        
        # Settings
dtype = pt.float32
device = pt.device("cpu")
#device = pt.device("cuda:0") # Uncomment this to run on GPU

def train(dataset_name):
    # Read parameters
    with open('config/'+dataset_name+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        fn_train, fn_out, hinge_resolution, gamma, q_resolution, t_steps = yaml_data['fn_train'], yaml_data['fn_out'], yaml_data['hinge_resolution'], yaml_data['gamma'], yaml_data['q_resolution'], yaml_data['t_steps']
        fn_train = os.getcwd() + fn_train
        fn_out = os.getcwd() + fn_out
        
    # Read data
    for framei in range(t_steps-1):
        # Load data
        print('\nReading ' + fn_train + '.csv...'.format(framei))
        df = pd.read_csv(fn_train + '.csv'.format(framei), delimiter=',').values[:, :]
        df = df[df[:,0] == framei, 1:]

        # Determine the lidar area
        delta = 1
        area_min, area_max = df[:, :2].min(axis=0), df[:, :2].max(axis=0)
        area_max_min = [area_min[0] - delta, area_max[0] + delta, area_min[1] - delta, area_max[1] + delta]
        print('Cell area = {}'.format(area_max_min))

        # Prepare data
        df = pt.tensor(df, dtype=pt.float32)
        X = df[:, :2]
        y = df[:, :2].reshape(-1, 1)
        print(' Data shape={}'.format(X.shape))

        # Define the model
        bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=None, cell_resolution=hinge_resolution, cell_max_min=area_max_min, X=X, nIter=1)

        # Fit the model
        t1 = time.time()
        bhm_mdl.fit(X, y)
        t2 = time.time()
        trainTime = np.round(t2-t1, 3)
        print(' Training time={} s'.format(trainTime))

        # Query the model
        xx, yy= np.meshgrid(np.arange(area_max_min[0], area_max_min[1] + q_resolution, q_resolution),
                             np.arange(area_max_min[2], area_max_min[3] + q_resolution, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        Xq = pt.tensor(grid, dtype=pt.float32)
        t1 = time.time()
        yq = bhm_mdl.predict(Xq)
        t2 = time.time()
        queryTime = np.round(t2-t1, 3)
        print(' Query time={} s'.format(queryTime))
        
        #update chart
        
        #run chart
        
        # Plot frame i
        plt.close('all')
        plt.figure(figsize=(3,3))
        plt.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=20, vmin=0, vmax=1, edgecolors='')
        plt.colorbar()
        plt.title('frame{}'.format(framei))
        #plt.savefig(fn_out+'_frame{}.png'.format(framei))
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.4,0.8])
        plot = ax.scatter         #sys.exit()
        print(' Plotted.')
        np.save(fn_out+'_frame{}'.format(framei), np.hstack((Xq,yq[:,None])),allow_pickle=True, fix_imports=True) #save files as numpy array
      
    #sys.exit()
    
#def update_map():
   

    #map_.set_array(intens)
    #time.sleep(1)
    #return map_,

def update_line(num, iterator, line):
    fn = next(iterator)
    print('output/real_lidar/' + fn)
    data = np.load('output/real_lidar/' + fn, allow_pickle=True) #load file
    line.set_offsets(data[:, :2])
    line.set_color(cm.jet(data[:, 2]))
    return line,
    

def run():
    #make animation
    #lidar = RPLidar(PORT_NAME)
    fig = plt.figure()
    #ax = plt.subplot(111)
    ax = plt.axes(xlim=(-5,5), ylim=(-5,5))
    #line = ax.scatter([0, 0], [0, 0], s=5, 
                           #cmap=plt.cm.Greys_r, lw=0)
    line = ax.scatter([0, 0], [0, 0], s=50, c=[IMIN, IMAX],
                         cmap=plt.cm.Greys_r, lw=0)
    
    #line = ax.scatter([0],[0],s=5)
    #line, = ax.plot([],[],'ro')
    #line, = ax.plot([], [], lw = 10)
    #line, = ax.plot([], lw = 3)
    #line = plt.scatter([],[])
                           
    
    #ax.set_rmax(DMAX)
    
    
    #line = ax.scatter(Xq[:, 0], Xq[:, 1])
    
    #ax.grid(True)

    path = 'output/real_lidar'
    bhmscans = os.listdir(path) #read names of numpy arrays (list of all file names in path)
    #print(bhmscans)
    iterator = iter(bhmscans)

    ani = FuncAnimation(fig, update_line,
        fargs=(iterator, line), interval=250)

    # ani = animation.FuncAnimation(map_, update_map,
    #                               fargs=(iterator), interval=50)
    
    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


    #ani.save('animation.mp4',writer=writer)

    plt.show()
    #lidar.stop()
    #lidar.disconnect()

if __name__ == '__main__':
    
    def __init__(self, canvas):
        self.canvas = canvas
    filename = 'examples'
    num_of_scans_to_stack = 10
    #main(filename, num_of_scans_to_stack)
    
    dataset_name = 'real_lidar_cut'
    #train(dataset_name)
    #update_line()
    run()
    
