@author: betsa
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from kerastuner import HyperParameter, HyperParameters
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner import Objective
from tensorflow import keras

# Dataset
X=tf.random.uniform(shape=(300,15,15,1),maxval=11,minval=4)
Y=tf.random.uniform(shape=(300,1),maxval=11,minval=4)
print(X.shape)
print(Y.shape)
#--------------------------------------------------
# Create model for KerasClassifier
def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [5,7,9]),
        #activation function
        activation='relu',
        input_shape=(15,15,1)),
    # adding second convolutional layer 
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        #activation function
        activation='relu'
    ),
    # adding flatten layer    
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(1, activation='linear')
    ])
    #compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])
    return model
#-----------------------------------------------------------------------
tuner = RandomSearch(build_model,
                    objective='val_mse',
                    max_trials =20
                     )
tuner.search(X, Y, epochs=10,validation_data=(X,Y))
best_hps = tuner.get_best_hyperparameters()[0]
# Show the results
print("======================================================")
print("Best:param:" , best_hps.values)
#tuner.results_summary()
best_model = tuner.get_best_models()[0]

best_model.summary()
