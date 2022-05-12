# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:14:05 2022

@author: betsa
"""

# this an XOR classification problem
import tensorflow as tf
import numpy as np
import matplotlib as plt
from mlxtend.plotting import plot_decision_regions

# 1-create a toy 200 random samples
np.random.seed(1)
x=np.random.uniform(low=-1, high=1, size=(200,2))
t=x[:,0] * x[:,1]
y=np.ones(200)
y[t<0]=0

# 2-select 100 for train and 100 for validation
ind=np.random.randint(low=0, high=200, size=100)
X_train=x[ind]
Y_train=y[ind]
ind2=tf.range(0,200)
ind2=ind2.numpy()
ind2=np.setdiff1d(ind2, ind)
X_valid=x[ind2]
Y_valid=y[ind2]

# 3-plot
'''fig=plt.pyplot.figure(figsize=(6,6))
plt.pyplot.plot(x[y==0,0],x[y==0,1],'o',markersize=10)
plt.pyplot.plot(x[y==1,0],x[y==1,1],'>',markersize=10)
plt.pyplot.xlabel('x1')
plt.pyplot.ylabel('x2')'''

# 4-model
mdl=tf.keras.Sequential()
mdl.add(tf.keras.layers.Dense(units=1,
                              activation=tf.keras.activations.sigmoid,
                              input_shape=(2,)))

mdl.compile(optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.BinaryCrossentropy(), 
            metrics=[tf.keras.metrics.BinaryAccuracy()] )     

hist = mdl.fit(X_train, Y_train,validation_data=(X_valid, Y_valid),epochs=200,batch_size=2,verbose=0)
results=hist.history

# 5-visualize the results
fig=plt.pyplot.Figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.pyplot.plot(results['loss'])
plt.pyplot.plot(results['val_loss'])
plt.pyplot.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax=fig.add_subplot(1,3,2)
plt.pyplot.plot(results['binary_accuracy'])
plt.pyplot.plot(results['val_binary_accuracy'])
plt.pyplot.legend(['Train ACC', 'Validation ACC'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=X_valid,y=Y_valid.astype(np.integer),clf=mdl)
