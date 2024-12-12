#######################################################################################################################
# Nuno Pessanha Santos, PhD
# 2024
# Note: Train Network - Translation & Orientation
#######################################################################################################################

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
# Define the Model
#######################################################################################################################

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
    
    # Output layer for X, Y, Z coordinates and quaternion components
    model.add(Dense(7, activation='linear'))
    
    return model

#######################################################################################################################
def create_model_3D_V3():
    model = Sequential()
    
    # Add Convolutional layers with BatchNormalization and LeakyReLU
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(3, 3, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))  # Dropout for regularization

    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers with Dropout and LeakyReLU
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output layer for X, Y, Z coordinates and quaternion components
    model.add(Dense(7, activation='linear'))
    
    return model


###############################################################################################################

def create_model_2D_points():
    model = Sequential()
    
    # Reshape input data to fit the Dense input shape
    model.add(Flatten(input_shape=(9, 2)))
    
    # Add Dense layers with BatchNormalization and Dropout
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # Add dropout to prevent overfitting
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # Add dropout to prevent overfitting
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))  # Add dropout to prevent overfitting
    
    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers with Dropout
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Add dropout to prevent overfitting
    
    # Output layer for X, Y, Z coordinates and quaternion components
    model.add(Dense(7, activation='linear'))
    
    return model

###############################################################################################################

def create_model_2D_points_V2():
    # Shared layers for feature extraction
    inputs = Input(shape=(3, 3, 2))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    flat = Flatten()(pool2)
    dense1 = Dense(128, activation='relu')(flat)
    dropout1 = Dropout(0.5)(dense1)

    # Translation branch
    dense_trans = Dense(64, activation='relu')(dropout1)
    dropout_trans = Dropout(0.5)(dense_trans)
    output_translation = Dense(3, activation='linear', name='translation')(dropout_trans)

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

    # Concatenate outputs
    output_combined = concatenate([output_translation, output_quaternion])

    model = Model(inputs=inputs, outputs=output_combined)

    return model

###############################################################################################################

def create_model_2D_points_V2_with_attention_and_stn():
    # Shared layers for feature extraction
    inputs = Input(shape=(3, 3, 2))

    # Self-Attention Mechanism
    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    attention = Multiply()([inputs, attention])

    # Spatial Transformer Network
    stn_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(attention)
    stn_pool1 = MaxPooling2D((2, 2), padding='same')(stn_conv1)
    stn_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(stn_pool1)
    stn_pool2 = MaxPooling2D((2, 2), padding='same')(stn_conv2)
    stn_flat = Flatten()(stn_pool2)
    stn_dense1 = Dense(128, activation='relu')(stn_flat)
    stn_dropout1 = Dropout(0.5)(stn_dense1)

    # Translation branch
    dense_trans = Dense(64, activation='relu')(stn_dropout1)
    dropout_trans = Dropout(0.5)(dense_trans)
    output_translation = Dense(3, activation='linear', name='translation')(dropout_trans)

    # Quaternion branch
    dense_quaternion = Dense(64, activation='relu')(stn_dropout1)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='tanh', name='quaternion')(dropout_quaternion)

    # Quaternion normalization to ensure unit length
    def normalize_quaternion(x):
        """
        Function to normalize quaternion to unit length.
        """
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    # Concatenate outputs
    output_combined = concatenate([output_translation, output_quaternion])

    model = Model(inputs=inputs, outputs=output_combined)

    return model

###############################################################################################################

def create_model_2D_points_V2_with_attention_and_stn_leakyrelu():
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

    dense_trans = Dense(64)(stn_dropout1)
    dense_trans = LeakyReLU(alpha=0.2)(dense_trans)
    dropout_trans = Dropout(0.5)(dense_trans)
    output_translation = Dense(3, activation='linear', name='translation')(dropout_trans)

    dense_quaternion = Dense(64)(stn_dropout1)
    dense_quaternion = LeakyReLU(alpha=0.2)(dense_quaternion)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='tanh', name='quaternion')(dropout_quaternion)

    def normalize_quaternion(x):
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    output_combined = concatenate([output_translation, output_quaternion])

    model = Model(inputs=inputs, outputs=output_combined)

    return model

###############################################################################################################

def create_modified_model():
    inputs = Input(shape=(3, 3, 2))

    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(inputs)
    attention = Multiply()([inputs, attention])

    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(attention)
    stn_pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(128, (3, 3), padding='same', activation='relu')(stn_pool1)
    stn_pool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    
    stn_flat = Flatten()(stn_pool2)
    stn_dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(stn_flat)
    stn_dropout1 = Dropout(0.5)(stn_dense1)

    dense_trans = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(stn_dropout1)
    dropout_trans = Dropout(0.5)(dense_trans)
    output_translation = Dense(3, activation='linear', name='translation')(dropout_trans)

    dense_quaternion = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(stn_dropout1)
    dropout_quaternion = Dropout(0.5)(dense_quaternion)
    output_quaternion_raw = Dense(4, activation='tanh', name='quaternion')(dropout_quaternion)

    def normalize_quaternion(x):
        return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))

    output_quaternion = Lambda(normalize_quaternion, name='normalized_quaternion')(output_quaternion_raw)

    output_combined = concatenate([output_translation, output_quaternion])

    model = Model(inputs=inputs, outputs=output_combined)

    return model

#######################################################################################################################
#######################################################################################################################
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

quaternion_loss_weight = 25  # Adjust this value as needed
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
    total_loss = translation_loss + quaternion_loss_value * quaternion_loss_weight
    
    return total_loss

#######################################################################################################################
#######################################################################################################################

# Compile the model
def compile_model(model):
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy','mean_absolute_error','mean_squared_error','mean_absolute_percentage_error','mean_squared_logarithmic_error', 'cosine_similarity','log_cosh'])

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


#######################################################################################################################
# Main
#######################################################################################################################

# Main function
def main():
    
    # Read data
    file_path = "Dataset.txt"  
    data = read_data(file_path)
    
    # Preprocessing
    X, X2, Y = preprocess_data_translation_orientation_quaternion(data)
    X2 = np.squeeze(X2, axis=1)  # Remove the extra dimension from X2

    #######################################################################################################################

    # Debug
    #print("The value X is:", X[0])
    #print("The value Y is:", Y[0])

    # Create Model
    #model = create_model_3D_V2()
    #model = create_model_2D_points()
    #model = create_model_2D_points_V2()
    #model = create_model_3D_V3()
    #model = create_model_2D_points_V2_with_attention_and_stn()
    #model = create_model_2D_points_V2_with_attention_and_stn_leakyrelu()
    model = create_modified_model()

    #plot_model(model, to_file='model.png', show_shapes=True)

    # Summary
    model.summary()

    # Compile model
    compile_model(model)
    
    # Train model
    epochs = 50000
    batch_size = 32
    history = train_model(model, X, Y, epochs, batch_size, 0.20)
    
    # Plot loss and accuracy history
    plot_loss_accuracy(history)

    # Save the trained model
    model.save('model_total_create_modified_model_50000.h5')
    
    # Save training history and model configuration
    file_path_results = "training_info_create_modified_model_50000.txt"  
    save_history_and_model(history, model, file_path_results)
    


# Entry point of the script
if __name__ == "__main__":
    main()