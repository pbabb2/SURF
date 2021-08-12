# Bayesian Hilbert Maps on TPU

* Setup
![Alt text](./SURF_setup.jpg)
* We use RPLidar
<!---
#![Alt text](./output/surf_patrick_toy_intersection/_frame0.png?raw=true "Regression Sample")
 -->
![Alt text](./SURFgif2.gif)

To convert rplidar raw data (offline) to BHM compatible csv, run  ```rplidar_to_bhm_convert_offline.py ```. Data will be saved in datasets (and datasets/figs/).
To run BHM, run  ```main_bhm_pytorch.py ```. Parameters of BHM can be set in the ```yaml``` files in the config folder.
To convert BHM compatible csv to an updatable Bayesian Hilbert Map, run 
```rplidar_to_bhm_convert_online.py```
To create Bayesian Hilbert Maps from the Lidar directly, run ```rplidar_to_bhm_live_fromlidar2.py```



## Requirements
- TensorFlow Lite
- PyTorch
- Matplotlib
- Pandas
- Numpy

## Installation 

- link websites for installation
- TensorFlow: `https://qengineering.eu/install-tensorflow-1.15.2-on-raspberry-pi-4.html`
- PyTorch: `https://gist.github.com/akaanirban/621e63237e63bb169126b537d7a1d979`

## Run

In the project directory, you can run:

### `rplidar_to_bhm_live_fromlidar2.py` 

To run the Edge TPU computer vision code go to the edge-tpu-tiny-yolo_ directory and run:

### `inference.py --model quant_coco-tiny-v3-relu_edgetpu.tflite --anchors tiny_yolo_anchors.txt --classes coco.names --cam -t 0.1 --edge_tpu --quant`


