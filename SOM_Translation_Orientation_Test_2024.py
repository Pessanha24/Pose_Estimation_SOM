# Nuno Pessanha Santos, PhD
# 2024
# Note: Test Network - Translation & Orientation

#######################################################################################################################
# Import
#######################################################################################################################

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation,Conv2D,Flatten, Add, GlobalAveragePooling2D, Reshape, Permute, Multiply, MaxPooling2D, concatenate, InputLayer, Lambda
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
from keras.layers import GaussianNoise
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
def preprocess_data_translation_orientation_quaternion(data):
    X = []
    X2 = []
    y = []
    for line in data:
        values = line.split()
        points = np.array([list(map(float, values[i:i+2])) for i in range(10, len(values), 2)])
        X.append(points.reshape(3, 3, 2))  # Reshape points to match expected input shape
        points_reshaped = np.reshape(points, (1, 9, 2))  # Reshape without squeezing
        X2.append(points_reshaped)
        y.append([float(val) for val in values[:3]] + [float(val) for val in values[6:10]])
    return np.array(X), np.array(X2), np.array(y)

#######################################################################################################################
# Model
#######################################################################################################################

# Predict with model
def predict_with_model(model, X):
    predictions = model.predict(X)
    return predictions

# Evaluate accuracy
def evaluate_accuracy(y_true, y_pred):
    acc = y_true - y_pred;
    return acc

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def quaternion_loss(y_true, y_pred):
    # Extract quaternion components from true and predicted outputs
    quaternion_true = y_true[:, 3:]
    quaternion_pred = y_pred[:, 3:]
    
    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(quaternion_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = quaternion_pred / quaternion_pred_norm
    
    # Quaternion dot product
    dot_product = tf.reduce_sum(quaternion_true * quaternion_pred_normalized, axis=-1, keepdims=True)
    
    # Compute quaternion loss
    quaternion_loss = 1 - tf.square(dot_product)
    
    return tf.reduce_mean(quaternion_loss)

# Custom loss function combining MSE for translation and Quaternion Loss for orientation
def custom_loss(y_true, y_pred):
    # Extract translation and quaternion components from true and predicted outputs
    translation_true = y_true[:, :3]
    translation_pred = y_pred[:, :3]
    
    # Compute Mean Squared Error for translation
    translation_loss = tf.reduce_mean(tf.square(translation_true - translation_pred))
    
    # Compute Quaternion Loss for orientation
    quaternion_loss_value = quaternion_loss(y_true, y_pred)
    
    # Combine the two losses
    total_loss = translation_loss + quaternion_loss_value
    
    return total_loss

#######################################################################################################################
# Main
#######################################################################################################################

# Main function
def main():
   
    # Read data
    file_path = "Data_2.txt"  
    data = read_data(file_path)

    # Preprocessing
    X, X2, Y = preprocess_data_translation_orientation_quaternion(data)
    
    # Load Model
    model = load_model('model_total.h5', custom_objects={'custom_loss': custom_loss})

    # Predict
    predictions = predict_with_model(model, X)
    
    # Accuracy
    accuracy = evaluate_accuracy(Y, predictions)
    
    print("Y:", Y)
    print("predictions:", predictions)
    print("Accuracy:", accuracy)



# Entry point of the script
if __name__ == "__main__":
    main()