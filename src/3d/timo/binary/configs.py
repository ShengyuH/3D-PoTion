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
from keras.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau,EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import backend as K # see the image_data_format information
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.models import Model,load_model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization
from sklearn.metrics import classification_report


config={}
config['data_format']=K.image_data_format()
config['num_classes']=15
config['kernel_initializer']='glorot_normal'
config['dropout_rate']=0.4
config['activation']='relu'
config['optimizer']='adam'
config['loss']='categorical_crossentropy'
config['epoch']=60
config['batch_size']=32
config['lr']=1e-4

config['model_name']='3D-32'
config['dir_potion']=os.path.join('../../potion',config['model_name'])
config['n_channels']=64
config['frame_size']=32
config['output']='./timo'