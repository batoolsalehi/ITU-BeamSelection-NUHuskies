########################################################
#Project name: ITU beam selection challenge
#Authors: NU Huskies team
#Date: 15/Oct/2020
########################################################
from __future__ import division

import os
import csv
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import random
from time import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest

from custom_metrics import *
############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.set_random_seed(seed)
#tf.random_set_seed()
np.random.seed(seed)
random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_model(model_flag, model,save_path):

    # save the model structure first
    model_json = model.to_json()
    print('\n*************** Saving New Model Structure ***************')
    with open(os.path.join(save_path, "%s_model.json" % model_flag), "w") as json_file:
        json_file.write(model_json)
        print("json file written")
        print(os.path.join(save_path, "%s_model.json" % model_flag))


# loading the model structure from json file
def load_model_structure(model_path='/scratch/model.json'):

    # reading model from json file
    json_file = open(model_path, 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    return model


def load_weights(model, weight_path = '/scratch/weights.02-3.05.hdf5'):
    model.load_weights(weight_path)


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape   # shape is (#,256)

        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        return y

def getBeamOutput(output_file):
    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)

    return y,num_classes

def custom_label(output_file, strategy='one_hot' ):
    'This function generates the labels based on input strategies, one hot, reg'

    print("Reading beam outputs...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)

    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y_shape = y.shape

    if strategy == 'one_hot':
        k = 1           # For one hot encoding we need the best one
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            max_index = logOut.argsort()[-k:][::-1]
            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y,num_classes


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str)

parser.add_argument('--json_file_path', help='Location of  json', type=str,default='/home/batool/beam_selection_NU/baseline_code/model_folder/test_model.json')
parser.add_argument('--hdf5_file_path', help='Location of weights', type=str,default = '/home/batool/beam_selection_NU/baseline_code/model_folder/best_weights.coord_img_lidar_custom.h5')


parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)

parser.add_argument('--image_feature_to_use', type=str ,default='v1', help='feature images to use',choices=['v1','v2','custom'])


args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

###############################################################################
# Inputs
###############################################################################

if 'coord' in args.input:
    #test
    X_coord_test = open_npz(args.test_data_folder + 'coord_input/coord_test.npz', 'coordinates')
    ### For convolutional input
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))

if 'img' in args.input:
    ###############################################################################
    #test
    X_img_test = open_npz(args.test_data_folder+'image_custom_input'+'/img_input_test_' + str(20) + '.npz','inputs')
    print('********************Reshape images for convolutional********************')
    X_img_test = X_img_test.reshape((X_img_test.shape[0], X_img_test.shape[1], X_img_test.shape[2],1))

if 'lidar' in args.input:
    ###############################################################################
    #test
    X_lidar_test = open_npz(args.test_data_folder + 'lidar_input/lidar_test.npz', 'input')

###############################################################################
# Load trained model
###############################################################################
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

with open(args.json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy])

model.load_weights(args.hdf5_file_path)


multimodal = False if len(args.input) == 1 else len(args.input)

if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        Y_test = model.predict([X_coord,X_lidar])

    elif 'coord' in args.input and 'img' in args.input:
        Y_test = model.predict([X_coord,X_img])
    else:
        Y_test = model.predict([X_lidar,X_img])
elif multimodal == 3:
    Y_test = model.predict([X_lidar_test,X_img_test,X_coord_test])
else:
    if 'coord' in args.input:
        Y_test = model.predict(X_coord)
    elif 'img' in args.input:
        Y_test = model.predict(X_img)
    else:
        Y_test = model.predict(X_lidar)
print(Y_test)

np.savetxt('beam_test_pred.csv', Y_test, delimiter=',')

# # Checking
# y_test, _ = custom_label('/home/batool/beam_selection_NU/baseline_code/test_front_end/itu_s009/baseline_data/beam_output/beams_output_test.npz','one_hot')
# print(y_test)

# def top_10_accuracy(y_true,y_pred):
#     return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)
# top1 = top_10_accuracy(y_test,Y_test)
# with tf.Session() as sess: print(top1.eval())
