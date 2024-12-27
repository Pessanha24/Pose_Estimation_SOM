# Nuno Pessanha Santos, PhD
# 2024
# Note: Train Network - Translation


#######################################################################################################################
# Import
#######################################################################################################################

from pyexpat import model
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
from keras.layers import GaussianNoise, LeakyReLU

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


#######################################################################################################################
# Define the Model
#######################################################################################################################

# The original Weights from SOM - 3x3 with 2 weigths - Try to retrieve information from their position
def create_model_3D():
    
    model = Sequential()
    
    # Add Convolutional layers with BatchNormalization
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(3, 3, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout for regularization

    # Add Gaussian noise layer
    #model.add(GaussianNoise(150))  
    
    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers with Dropout
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output layer
    model.add(Dense(3, activation='linear'))
    
    return model

#######################################################################################################################
# The original Weights from SOM - 3x3 with 2 weigths - Try to retrieve information from their position
def create_model_3D_V2():
    
    model = Sequential()
    
    # Add Convolutional layers with BatchNormalization
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(3, 3, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers with Dropout
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output layer
    model.add(Dense(3, activation='linear'))
    
    return model

#######################################################################################################################

def create_model_3D_V3():
    
    model = Sequential()
    
    # Add Convolutional layers with BatchNormalization
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(3, 3, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers with Dropout
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output layer
    model.add(Dense(3, activation='linear'))

    return model




# Compile the model
def compile_model(model):
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity'])


#######################################################################################################################
# Esoteric model
#######################################################################################################################

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

def create_state_of_the_art_model(): # USED WITH BEST RESULTS
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
##############################################################
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


##############################################################
##############################################################
##############################################################

def create_improved_model():
    input_layer = Input(shape=(3, 3, 2))

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

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def create_simplified_model():
    input_layer = Input(shape=(3, 3, 2))

    x = Flatten()(input_layer)
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(3, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# Define the model
def create_refined_model():
    input_layer = Input(shape=(3, 3, 2))
    
    # Add Convolutional layers with BatchNormalization and Squeeze-and-Excitation blocks
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)
    
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)
    
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)
    x = Dropout(0.5)(x)
    
    # Add Self-Attention block
    x = self_attention_block(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = self_attention_block(x)
    x = Dropout(0.5)(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=l2(0.001), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, kernel_regularizer=l2(0.001), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    output_layer = Dense(3, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def create_improved_model_similar_limpo():
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Dropout(0.5)(x)

    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
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
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def create_improved_model_similar_limpo_2():
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Dropout(0.5)(x)

    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
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
    #x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(3, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def create_improved_model_similar_limpo_3():
    input_layer = Input(shape=(3, 3, 2))

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(input_layer)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Dropout(0.5)(x)

    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = self_attention_block(x)
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
    #x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    #x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(3, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

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
# Main
#######################################################################################################################

# Main function
def main():
    
    # Read data
    file_path = "Dataset_Version_2.txt"  
    data = read_data(file_path)
    
    # Preprocessing
    X, Y = preprocess_data_3D(data)
    
    # Create Model
    #model = create_model_3D()
    #model = create_model_3D_V2()
    #model = create_model_3D_V3()
    #model = create_state_of_the_art_model()
    #model = create_improved_model()
    #model = create_simplified_model()
    #model = create_refined_model()
    
    model = create_improved_model_similar_()
    #model = create_improved_model_similar_limpo()
    #model = create_improved_model_similar_limpo_2()
    #model = create_improved_model_similar_limpo_3()
    

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
    file_path_results = "training_create_improved_model_similar_50000_256_.txt"  
    save_history_and_model(history, model, file_path_results)
    
    # Save the trained model
    model.save('model_create_improved_model_similar_50000_256_.h5') 

# Entry point of the script
if __name__ == "__main__":
    main()
