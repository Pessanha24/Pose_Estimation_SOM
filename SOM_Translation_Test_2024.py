import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation,Conv2D,Flatten, Add, GlobalAveragePooling2D, Reshape, Permute, Multiply
from keras.activations import tanh
from keras.models import Model, Sequential,load_model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import Huber, LogCosh
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.utils import register_keras_serializable
import joblib
import netron

# Read file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Data pre-processing - y = [X,Y,Z]
def preprocess_data_3D(data):
    X = []
    y = []
    for line in data:
        values = line.split()
        points = np.array([list(map(float, values[i:i+2])) for i in range(10, len(values), 2)])
        X.append(points.reshape(3, 3, 2))  # Reshape points to match expected input shape
        y.append([float(val) for val in values[:3]])  # First 3 numbers
    return np.array(X), np.array(y)

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_mean, noise_std):
    noise = np.random.normal(noise_mean, noise_std, X.shape)
    X_noisy = X + noise
    return X_noisy

def squeeze_excite_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]

    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)

    se = Reshape((1, 1, channels))(se)
    scaled_input = Multiply()([input_tensor, se])

    return scaled_input

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

def create_state_of_the_art_model():
    input_layer = Input(shape=(3, 3, 2))

    # Add Convolutional layers with BatchNormalization and Squeeze-and-Excitation blocks
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.5)(x)

    # Add Self-Attention block
    x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Dropout(0.5)(x)
    
    # Add Convolutional layers with BatchNormalization
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

    # Flatten layer
    x = Flatten()(x)
    
    # Add Dense layers with Dropout
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output_layer = Dense(3, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

##############################################################

def create_improved_model_similar_():
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Dropout(0.5)(x)

    x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = self_attention_block(x)
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
    output_layer = Dense(3, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Predict with model
def predict_with_model(model, X):
    predictions = model.predict(X)
    return predictions

# Evaluate accuracy
def evaluate_accuracy(y_true, y_pred):
    acc = y_true - y_pred
    return acc

# Main function
def main():
    # Read data
    file_path = "5_meters_0.5_Final.txt"
    data = read_data(file_path)

    # Preprocessing
    X, Y = preprocess_data_3D(data)
    
    mean_ = 0
    #variance_ = 150
    #standard_deviation_ = sqrt(variance_)
    standard_deviation_ = 30;
    X2 = add_gaussian_noise(X,mean_,standard_deviation_)

    # Create model
    #model = create_state_of_the_art_model()
    model = create_improved_model_similar_()

    # Load weights
    try:
        model.load_weights('model_create_improved_model_similar_50000_data_2_256.h5')
    except Exception as e:
        print("Error loading weights:", e)
        return

    # Predict
    predictions = predict_with_model(model, X2)
    
    # Accuracy
    accuracy = evaluate_accuracy(Y, predictions)
    
    np.savetxt('accuracy_Translation_5_meters_0.5_Final_mean_0_std_30.txt', accuracy, delimiter=',', fmt='%f')
    np.savetxt('predictions_Translation_5_meters_0.5_Final_mean_0_std_30.txt', predictions, delimiter=',', fmt='%f')

    
    print("Y:", Y)
    print("predictions:", predictions)
    print("Accuracy:", accuracy)

    # Run Netron
    #netron.start('model_create_state_of_the_art_model_30000_256.h5')

# Entry point of the script
if __name__ == "__main__":
    main()
