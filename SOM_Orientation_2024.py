# Nuno Pessanha Santos, PhD
# 2024
# Note: Train Network - Orientation

# export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib

# cd /media/pessanha/Data_6_HDD3/SOM_Translation_2024

#######################################################################################################################
# Import
#######################################################################################################################

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation,Conv2D,Flatten, Add, GlobalAveragePooling2D, Reshape, Permute, Multiply, MaxPooling2D, concatenate, InputLayer, Lambda, Dot
from keras.activations import tanh
from keras.models import Model, Sequential,load_model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import Huber, LogCosh
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.utils import register_keras_serializable, plot_model, model_to_dot
import joblib
from keras.layers import GaussianNoise, LeakyReLU
import matplotlib.pyplot as plt
from IPython.display import Image
import pydotplus

#######################################################################################################################
# Read dataset
#######################################################################################################################

# Read file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Data pre-processing - y = [X,Y,Z,Alpha,Beta,Gamma]
def preprocess_data_6D(data):
    X = []
    y = []
    for line in data:
        values = line.split()
        points = np.array([list(map(float, values[i:i+2])) for i in range(10, len(values), 2)])
        X.append(points.reshape(3, 3, 2))  # Reshape points to match expected input shape
        y.append([float(val) for val in values[:6]])  # First 6 numbers
    return np.array(X), np.array(y)

# Data pre-processing - y = [Alpha,Beta,Gamma]
def preprocess_data_orientation_Euler(data):
    X = []
    y = []
    for line in data:
        values = line.split()
        points = np.array([list(map(float, values[i:i+2])) for i in range(10, len(values), 2)])
        X.append(points.reshape(3, 3, 2))  # Reshape points to match expected input shape
        y.append([float(val) for val in values[3:6]])  # First 6 numbers
    return np.array(X), np.array(y)

# Data pre-processing - y = [q_w,q_x,q_y,q_z]
def preprocess_data_orientation_quaternion(data):
    X = []
    y = []
    for line in data:
        values = line.split()
        points = np.array([list(map(float, values[i:i+2])) for i in range(10, len(values), 2)])
        X.append(points.reshape(3, 3, 2))  # Reshape points to match expected input shape
        y.append([float(val) for val in values[6:10]])  # First 6 numbers
    return np.array(X), np.array(y)

#######################################################################################################################
# Define the Model
#######################################################################################################################

def create_model_2D_points_V2_quaternion():
    
    # Shared layers for feature extraction
    inputs = Input(shape=(3, 3, 2))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(128, activation='relu')(flat)
    dropout1 = Dropout(0.5)(dense1)

    # Quaternion branch
    dense_quaternion = Dense(64, activation='relu')(dropout1)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='tanh', name='quaternion')(dropout_quaternion)

    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    model = Model(inputs=inputs, outputs=output_quaternion)

    return model


###############################################################################################################

def create_model_2D_points_V2_with_attention_and_stn_leakyrelu_quaternion():
    inputs = Input(shape=(3, 3, 2))

    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    attention = Multiply()([inputs, attention])

    conv1 = Conv2D(32, (3, 3), padding='same')(attention)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    stn_pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(64, (3, 3), padding='same')(stn_pool1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    stn_pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    
    stn_flat = Flatten()(stn_pool2)
    stn_dense1 = Dense(128)(stn_flat)
    stn_dense1 = LeakyReLU(alpha=0.2)(stn_dense1)
    stn_dropout1 = Dropout(0.5)(stn_dense1)

    dense_quaternion = Dense(64)(stn_dropout1)
    dense_quaternion = LeakyReLU(alpha=0.2)(dense_quaternion)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='tanh', name='quaternion')(dropout_quaternion)

    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    model = Model(inputs=inputs, outputs=output_quaternion)

    return model

###############################################################################################################

def create_modified_model_quaternion():
    inputs = Input(shape=(3, 3, 2))

    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    attention = Multiply()([inputs, attention])

    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(attention)
    stn_pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(stn_pool1)
    stn_pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    
    stn_flat = Flatten()(stn_pool2)
    stn_dense1 = Dense(256, activation='relu')(stn_flat)
    stn_dropout1 = Dropout(0.5)(stn_dense1)

    dense_quaternion = Dense(128, activation='relu')(stn_dropout1)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='linear', name='quaternion')(dropout_quaternion)

    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    model = Model(inputs=inputs, outputs=output_quaternion)

    return model


###############################################################################################################

def create_modified_model_2_quaternion():
    # Input layer for 3x3 grid of 2D points
    inputs = Input(shape=(3, 3, 2))

    # Attention mechanism
    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    attention = Multiply()([inputs, attention])

    # Convolutional layers
    conv1 = Conv2D(32, (3, 3), padding='same')(attention)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    stn_pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(64, (3, 3), padding='same')(stn_pool1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    stn_pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    
    # Flatten and dense layers
    stn_flat = Flatten()(stn_pool2)
    stn_dense1 = Dense(128)(stn_flat)
    stn_dense1 = LeakyReLU(alpha=0.2)(stn_dense1)
    stn_dropout1 = Dropout(0.5)(stn_dense1)

    # Quaternion prediction layers
    dense_quaternion = Dense(64)(stn_dropout1)
    dense_quaternion = LeakyReLU(alpha=0.2)(dense_quaternion)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='linear', name='quaternion')(dropout_quaternion)

    # Normalize quaternion
    def normalize_quaternion(x):
        return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
    
    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    # Constructing the model
    model = Model(inputs=inputs, outputs=output_quaternion)
    
    return model

###############################################################################################################
###############################################################################################################
###############################################################################################################

def quaternion_loss(y_true, y_pred):
    # Ensure unit length of true quaternions
    quaternion_true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True))
    quaternion_true_normalized = y_true / quaternion_true_norm

    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = y_pred / quaternion_pred_norm

    # Quaternion dot product
    dot_product = tf.reduce_sum(quaternion_true_normalized * quaternion_pred_normalized, axis=-1)

    # Compute symmetric quaternion loss
    quaternion_loss = 1 - tf.square(tf.abs(dot_product))

    return tf.reduce_mean(quaternion_loss)

def geodesic_loss(y_true, y_pred):
    # Ensure unit length of true quaternions
    quaternion_true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True))
    quaternion_true_normalized = y_true / quaternion_true_norm

    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = y_pred / quaternion_pred_norm

    # Quaternion dot product
    dot_product = tf.reduce_sum(quaternion_true_normalized * quaternion_pred_normalized, axis=-1)

    # Compute geodesic loss
    clipped_dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)  # Clip dot product values for numerical stability
    geodesic_distance = tf.acos(clipped_dot_product)
    
    return tf.reduce_mean(geodesic_distance)

def quaternion_loss_2(y_true, y_pred):
    # Extract entire quaternions from true and predicted outputs
    quaternion_true = y_true
    quaternion_pred = y_pred
    
    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(quaternion_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = quaternion_pred / quaternion_pred_norm
    
    # Quaternion dot product
    dot_product = tf.reduce_sum(quaternion_true * quaternion_pred_normalized, axis=-1, keepdims=True)
    
    # Compute quaternion loss
    quaternion_loss = 1 - tf.square(dot_product)
    
    # Add regularization term to penalize large deviations
    regularization_loss = tf.reduce_mean(tf.square(quaternion_true - quaternion_pred))
    
    # Combine quaternion loss and regularization loss
    total_loss = tf.reduce_mean(quaternion_loss) + regularization_loss
    
    return total_loss

#######################################################################################################################

def QReLU(x):
    # Split the quaternion into real and imaginary parts
    real_part = x[..., 0]
    imaginary_parts = x[..., 1:]

    # Apply ReLU to the real part
    real_part_relu = tf.nn.relu(real_part)

    # Combine the real part with the original imaginary parts
    return tf.concat([real_part_relu[..., tf.newaxis], imaginary_parts], axis=-1)

def self_attention_block(input_tensor):
    channels = input_tensor.shape[-1]

    query = Conv2D(channels // 8, kernel_size=1)(input_tensor)
    key = Conv2D(channels // 8, kernel_size=1)(input_tensor)
    value = Conv2D(channels, kernel_size=1)(input_tensor)

    attention = Conv2D(1, kernel_size=1)(query)
    attention = Permute((3, 1, 2))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 3, 1))(attention)

    scaled_value = Multiply()([value, attention])

    return scaled_value

def create_improved_model_quaternion():
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output_quaternion_raw = Dense(4, activation='linear', name='quaternion')(x)  # Output quaternion
    
    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)
    
    model = Model(inputs=input_layer, outputs=output_quaternion)
    
    return model

def create_improved_model_quaternion_QRELU(): #"Quaternion Recurrent Neural Networks" by Palangi et al. (2019)
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Dropout(0.5)(x)
    output_quaternion_raw = Dense(4, name='quaternion')(x)  # Output quaternion
    
    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)
    
    model = Model(inputs=input_layer, outputs=output_quaternion)
    
    return model
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def create_improved_model_quaternion_QRELU_2(): #"Quaternion Recurrent Neural Networks" by Palangi et al. (2019)
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Dropout(0.5)(x)
    output_quaternion_raw = Dense(4, name='quaternion')(x)  # Output quaternion
    
    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)
    
    model = Model(inputs=input_layer, outputs=output_quaternion)
    
    return model
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def create_improved_model_quaternion_QRELU_3(): #"Quaternion Recurrent Neural Networks" by Palangi et al. (2019)
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dense(512, kernel_regularizer=l2(0.001))(x)
    #x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = Dense(256, kernel_regularizer=l2(0.001))(x)
    #x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output_quaternion_raw = Dense(4, name='quaternion')(x)  # Output quaternion
    
    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)
    
    model = Model(inputs=input_layer, outputs=output_quaternion)
    
    return model
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def create_improved_model_quaternion_QRELU_4(): #"Quaternion Recurrent Neural Networks" by Palangi et al. (2019)
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Conv2D(128, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)  # Apply self-attention
    x = Dropout(0.5)(x)
    
    x = Conv2D(256, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dense(512, kernel_regularizer=l2(0.001))(x)
    #x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = Dense(256, kernel_regularizer=l2(0.001))(x)
    #x = Activation(QReLU)(x)  # Apply QReLU activation
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output_quaternion_raw = Dense(4, name='quaternion')(x)  # Output quaternion
    
    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)
    
    model = Model(inputs=input_layer, outputs=output_quaternion)
    
    return model

#######################################################################################################################
#######################################################################################################################

# Compile the model
def compile_model(model):
    model.compile(loss=quaternion_loss, optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity'])
    #model.compile(loss=geodesic_loss, optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity'])
    #model.compile(loss=quaternion_loss_2, optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity'])


#######################################################################################################################
# Graphics & Datalogging
#######################################################################################################################

# Plot loss and accuracy history
def plot_loss_accuracy(history):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['accuracy'], label='accuracy')
    plt.title('Loss and Accuracy over Epochs')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
# Save training history and model configuration to a text file
def save_history_and_model(history, model, file_path):
    with open(file_path, 'w') as file:
        file.write("Training History:\n")
        for key, value in history.items():
            file.write(f"{key}: {value}\n")

        file.write("\nModel Configuration:\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        
# Train the model
def train_model(model, X_train, y_train, epochs_, batch_size_,validation_split_percentage):
    history = model.fit(X_train, y_train, validation_split=validation_split_percentage, epochs=epochs_, batch_size=batch_size_, verbose=1)
    return history.history

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def visualize_model(model, filename):
    # Get the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    # Plot the model architecture
    plt.figure(figsize=(12, 8))
    plt.text(0.1, 0.9, model_summary, {'fontsize': 10}, fontfamily='monospace')
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

######################################################################################################################
# Main
#######################################################################################################################

# Main function
def main():
    # Read data
    file_path = "Dataset_Version_2.txt"  
    data = read_data(file_path)
    
    # Preprocessing
    #X, Y = preprocess_data_orientation_Euler(data)
    X, Y = preprocess_data_orientation_quaternion(data)
    
    # Debug
    #print("The value X is:", X[0])
    #print("The value Y is:", Y[0])
    
    # Create Model
    #model = create_model_2D_points_V2_quaternion() #NAN
    #model = create_model_2D_points_V2_with_attention_and_stn_leakyrelu_quaternion() #NAN
    #model = create_modified_model_quaternion()
    #model = create_modified_model_2_quaternion()
    #model = create_improved_model_quaternion() #Similar translation
    model=create_improved_model_quaternion_QRELU()
    #model=create_improved_model_quaternion_QRELU_2()
    #model=create_improved_model_quaternion_QRELU_3()
    #model = create_improved_model_quaternion_QRELU_4()

    model.summary()
    
    # Compile model
    compile_model(model)
    
    # Train model
    epochs = 50000
    batch_size = 256
    history = train_model(model, X, Y, epochs, batch_size, 0.2)
    
    # Plot loss and accuracy history
    plot_loss_accuracy(history)
    
    # Save training history and model configuration
    file_path_results = "training_create_improved_model_quaternion_QRELU_quaternion_loss_data_2_50000_256_.txt"  
    save_history_and_model(history, model, file_path_results)
    
    # Save the trained model
    model.save('model_create_improved_model_quaternion_QRELU_quaternion_loss_data_2_50000_256_.h5') 

# Call the main function
if __name__ == "__main__":
    main()
