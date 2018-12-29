#----------------------------------------
# Function: Define parameters of tf_DUNet
# Author:   Changzhi Luo
# Date:     20180908
#---------------------------------------
import numpy as np

param = {
    # Images to use per iter
    'batch_size' : 8,
    # Maximum iterations of training
    'max_iters' : 200000,
    # Initial learning rate
    'init_lr' : 0.0002,
    # Steps to reduce learning rate
    'stepsize' : 2000, 
    # Decay rate of learning rate
    'decay_rate' : 0.9,
    # Directory of train/val/test images
    #'image_dir' : '../../data_mnist_9',
    # Directory to save models
    'model_dir' : '../../model/exp45',
    # Directory to save output data
    'output_dir' : '../../output/exp45',
    # Directory to save training logs
    'train_log_dir' : '../../log/train/exp45',
    # Directory to save summary information used for visualization
    'tensorboard_log_dir' : '../../log/tensorboard/exp45',
    # Loss weight of RNet
    'weight_R' : 500, 
    # Loss weight of ENet
    'weight_E' : 1,
    # Loss weight of DNet
    'weight_D' : 1,
    # Loss weight of cls_net
    'weight_C' : 1,
    # Interval to save a trained model
    'snap_interval' : 50000,
    # Maximum number of models to keep
    'max_to_keep' : 3,
    # Interval to save summary information
    'summary_interval' : 20,
    # GPU Device used for training/testing,which can be '0', '0, 1', '0, 1, 2', etc.
    'gpu_num' : '3',
    # Define per_process_gpu_memory_fraction.
    'gpu_memory_fraction' : 0.98,
    # Pixel mean values used for normalization. (1, 1, 3) array with an order of BGR
    'pixel_means' : np.array([[[102.9801, 115.9465, 122.7717]]]),
    # Normalized image height
    'load_height' : 256,
    # Normalized image width
    'load_width' : 256,
    # Flag to denote whether train from scratch or train from an intermediate model.
    'restore_train' : False, 
    # ng sample ratio interval in oversampling strategy
    'ng_ratio_low': 0.25,
    'ng_ratio_high': 0.5, 

    # The following parameters are used in testing phase only
    # Specify the model to test
    'num_test_iters': '200000'
}


