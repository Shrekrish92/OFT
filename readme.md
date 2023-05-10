# Orthographic Feature Transform for Monocular 3D Object Detection

![OFTNet-Architecture](https://github.com/tom-roddick/oft/raw/master/architecture.png "OFTNet-Architecture")
This is a PyTorch implementation of the OFTNet network from the paper [Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/abs/1811.08188). The code currently supports training the network from scratch on the KITTI dataset - intermediate results can be visualised using Tensorboard. This version of the project currrrently does not work on multiclass object detection. It only detects cars in the kitti dataset. This project does not use Lidar data to detect objects. Note also that there are some slight implementation differences from the original code used in the paper and the below cited source code.
[Source code](https://github.com/tom-roddick/oft/)

## Training
The training script can be run by calling `train.py` with the name of the experiment as a required position argument. 
```
python train.py name-of-experiment --gpu 0
```
By default data will be read from `data/kitti/objects` and model checkpoints will be saved to `experiments`. The model is trained using the KITTI 3D object detection benchmark which can be downloaded from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). See `train.py` for a full list of training options.

## Inference
To decode the network predictions and visualise the resulting bounding boxes, run the `infer.py` script with the path to the model checkpoint you wish to visualise:
```
python infer.py --model-path /path/to/checkpoint.pth.gz --gpu 0
```

## References
OFT 2023 Recoil[Source code](https://github.com/tom-roddick/oft/)

@article{roddick2018orthographic,  
  title={Orthographic feature transform for monocular 3d object detection},  
  author={Roddick, Thomas and Kendall, Alex and Cipolla, Roberto},  
  journal={British Machine Vision Conference},  
  year={2019}  
}

