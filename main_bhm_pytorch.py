# Bayesian Hilberct Maps for Patrick Babb

import sys
import os
import time
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pl
from bhmtorch_cpu import BHM2D_PYTORCH

def load_parameters(case):
    parameters = \
        {
         'surf_patrick': \
             (os.getcwd() + '/datasets/surf_patrick_toy_intersection', # data file
              os.getcwd() + '/output/surf_patrick_toy_intersection/', #output images
              (1, 1),  # hinge point resolution
              5,  # gamma - a hyperparameter
              0.9,  # query_resolution --> units: meters, every 0.5,....meters (smaller is better resolution
              ),
         'real_lidar_cut':  
             (os.getcwd() + '/datasets/real_lidar_cut', # data file
              os.getcwd() + '/output/real_lidar/', #output images
              (1000, 1000),  # hinge point resolution
              5,  # gamma - a hyperparameter
              0.9,  # query_resolution --> units: meters, every 0.5,....meters (smaller is better resolution
              ),
         }
    return parameters[case]

# Settings
dtype = pt.float32
device = pt.device("cpu")
#device = pt.device("cuda:0") # Uncomment this to run on GPU

def train():
    # Read parameters
    fn_train, fn_out, cell_resolution, gamma, q_resolution = load_parameters('real_lidar_cut') #call name depending on want

    # Read data
    for framei in range(10):
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
        y = df[:, 2].reshape(-1, 1)
        print(' Data shape={}'.format(X.shape))

        # Define the model
        bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=area_max_min, X=X, nIter=1)

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

        # Plot frame i
        pl.close('all')
        pl.figure(figsize=(3,3))
        pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=20, vmin=0, vmax=1, edgecolors='')
        pl.colorbar()
        pl.title('frame{}'.format(framei))
        pl.savefig(fn_out+'_frame{}.png'.format(framei))
        #sys.exit()
        print(' Plotted.')
    sys.exit()

if __name__ == '__main__':
    train()