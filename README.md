# Bayesian Hilbert Maps on TPU
* We use RPLidar

![Alt text](./output/surf_patrick_toy_intersection/_frame0.png?raw=true "Regression Sample")

To convert rplidar raw data (offline) to BHM compatible csv, run  ```rplidar_to_bhm_convert_offline.py ```. Data will be saved in datasets (and datasets/figs/).
To run BHM, run  ```main_bhm_pytorch.py ```. Parameters of BHM can be set in the ```yaml``` files in the config folder.



#necessary libraries

#install using pip3

$ pip install rplidar

$ sudo pip3 install rplidar


## Requirements
- Ruby version 2.6.0 (check your Ruby version using ruby -v)

## Installation 

Need to place these [steps from here](https://jekyllrb.com/docs/installation/) here.

## Available Scripts

In the project directory, you can run:

### `gem install jekyll bundler`

Installs all the Jekyll specific gems (software libraries)<br />
See the [setup section](https://jekyllrb.com/docs/step-by-step/01-setup/) in official Jekyll Docs for more setup information.
