from math import sqrt
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
import msvcrt

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

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_mean, noise_std):
    noise = np.random.normal(noise_mean, noise_std, X.shape)
    X_noisy = X + noise
    return X_noisy


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
#######################################################################################################################


# Predict with model
def predict_with_model(model, X):
    predictions = model.predict(X)
    return predictions

# Evaluate accuracy
def evaluate_accuracy(y_true, y_pred):
    # Ensure the input tensors are of the same type (float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Ensure unit length of true quaternions
    quaternion_true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True))
    quaternion_true_normalized = y_true / quaternion_true_norm

    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = y_pred / quaternion_pred_norm

    # Quaternion dot product
    dot_product = tf.reduce_sum(quaternion_true_normalized * quaternion_pred_normalized, axis=-1)
    
    # Ensure the dot product is within the valid range for acos due to numerical issues
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
    
    # Compute the angle between the two quaternions in radians
    angular_distance_radians = 2 * tf.acos(tf.abs(dot_product))
    
    # Convert radians to degrees
    angular_distance_degrees = angular_distance_radians * (180.0 / tf.constant(3.141592653589793))  # Conversion factor: 180/pi

    return angular_distance_degrees

def evaluate_accuracy_2(y_true, y_pred):
    # Ensure the input tensors are of the same type (float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Ensure unit length of true quaternions
    quaternion_true_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=-1, keepdims=True))
    quaternion_true_normalized = y_true / quaternion_true_norm

    # Ensure unit length of predicted quaternions
    quaternion_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1, keepdims=True))
    quaternion_pred_normalized = y_pred / quaternion_pred_norm

    # Calculate the conjugate of predicted quaternions
    quaternion_pred_conjugate = tf.concat([quaternion_pred_normalized[..., :1], -quaternion_pred_normalized[..., 1:]], axis=-1)

    # Quaternion multiplication: true * conjugate(pred)
    w1, x1, y1, z1 = tf.unstack(quaternion_true_normalized, axis=-1)
    w2, x2, y2, z2 = tf.unstack(quaternion_pred_conjugate, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    quaternion_error = tf.stack([w, x, y, z], axis=-1)

    # Compute the angle between the two quaternions in radians using the scalar part
    angular_distance_radians = 2 * tf.acos(tf.clip_by_value(tf.abs(quaternion_error[..., 0]), -1.0, 1.0))

    # Convert radians to degrees
    angular_distance_degrees = angular_distance_radians * (180.0 / tf.constant(3.141592653589793))  # Conversion factor: 180/pi

    return angular_distance_degrees


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


# Main function
def main():
    # Read data
    file_path = "5_meters_0.5_Final.txt"
    data = read_data(file_path)

    # Preprocessing
    X, Y = preprocess_data_orientation_quaternion(data)
    
    mean_ = 0
    #variance_ = 150
    #standard_deviation_ = sqrt(variance_)
    standard_deviation_ = 200;
    X2 = add_gaussian_noise(X,mean_,standard_deviation_)

    #print("X:", X)
    #print("X2:", X2)
    #msvcrt.getch()  # Wait for a key press

    # Create model
    model = create_improved_model_quaternion_QRELU()
    #model = create_improved_model()

    # Load weights
    try:
        model.load_weights('model_create_improved_model_quaternion_QRELU_quaternion_loss_data_2_50000_256.h5')
    except Exception as e:
        print("Error loading weights:", e)
        return

    # Predict
    predictions = predict_with_model(model, X2)
    
    # Accuracy
    #accuracy = evaluate_accuracy(Y, predictions)
    accuracy = evaluate_accuracy_2(Y, predictions)
    
    np.savetxt('accuracy_Orientation_5_meters_0.5_Final_mean_0_std_200.txt', accuracy, delimiter=',', fmt='%f')
    np.savetxt('predictions_Orientation_5_meters_0.5_Final_mean_0_std_200.txt', predictions, delimiter=',', fmt='%f')
    
    print("Y:", Y)
    print("predictions:", predictions)
    print("Accuracy:", accuracy)

    # Run Netron
    #netron.start('model_create_state_of_the_art_model_30000_256.h5')

# Entry point of the script
if __name__ == "__main__":
    main()