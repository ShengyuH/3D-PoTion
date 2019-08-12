import os, sys
import numpy as np
import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.image as mpimg
import imageio
import h5py
import random
from PIL import Image
import json
from astropy.modeling.models import Gaussian2D
from numpy.random import randint
import random

from keras.utils import to_categorical,Sequence
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Activation,Average,Concatenate
from keras.layers import Conv2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K # see the image_data_format information
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.models import Model, load_model
from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from sklearn.metrics import classification_report

from numpy.random import seed    # fix random seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

config={}
####################
# training settings
####################
config['data_format']=K.image_data_format()
config['num_classes']=15
config['n_filters']=[32,32,64,64]
config['strides']=[2,1,2,1,2,1,2,1,]
config['kernel_size']=3
config['kernel_initializer']='glorot_uniform'
config['dropout_rate']=0.0
config['activation']='relu'
config['optimizer']='adam'
config['loss']='categorical_crossentropy'
config['epoch']=100
config['batch_size']=32
config['lr']=1e-4


#####################
# IO settings
#####################
config['model_name']='gt-32'
config['dir_potion']=os.path.join('../../potion',config['model_name'])
config['n_channels']=52
config['frame_size']=32
config['output']='./output'