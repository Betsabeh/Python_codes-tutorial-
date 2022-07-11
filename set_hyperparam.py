import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from sklearn.model_selection import RandomizedSearchCV

# Load the dataset
data=pd.read_csv('iris.csv')
X=data.iloc[:,[0,1,2,3]].values
Y=data.iloc[:,[4]].values
print(len(Y))
Label=np.zeros(len(Y))
print(len(Label))
i=0
for item in Y:
  if item=='Setosa':
    Label[i]=1
  if item=='Versicolor':
    Label[i]=2
  if item=='Virginica':
    Label[i]=3
  i=i+1
print(Label)     


# Create model for KerasClassifier
def create_model(num_hidden=1,
                 num_unit=10,
                 activation='relu',
                 Drop_rate=0.5):
    # Model definition
    mdl=tf.keras.Sequential()
    for i in range(1,num_hidden):
      mdl.add(tf.keras.layers.Dense(units=num_unit,activation=activation))
      mdl.add(tf.keras.layers.Dropout(rate=Drop_rate))
    
    mdl.add(tf.keras.layers.Dense(units=1,activation=activation))
    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['accuracy'])
    return mdl



model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model) 

# Define the range
num_hidden = [2,4,6,8]
num_unit =[8,16,32]
activation=['elu', 'relu','linear']
Drop_rate = [0.2,0.5,0.7]

# Prepare the Grid
param_grid = dict(num_hidden=num_hidden, 
                  num_unit=num_unit, 
                  activation=activation,
                  Drop_rate=Drop_rate)

# GridSearch in action
grid = RandomizedSearchCV(model,param_grid,n_jobs=4,n_iter=20)
grid_result = grid.fit(X, Label)

# Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))