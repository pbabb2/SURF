# Bayesian Hilbert Maps on TPU
* We use RPLidar

![Alt text](./output/surf_patrick_toy_intersection/_frame0.png?raw=true "Regression Sample")

To convert rplidar raw data (offline) to BHM compatible csv, run  ```rplidar_to_bhm_convert_offline.py ```. Data will be saved in datasets (and datasets/figs/).
To run BHM, run  ```main_bhm_pytorch.py ```. Parameters of BHM can be set in the ```yaml``` files in the config folder.

