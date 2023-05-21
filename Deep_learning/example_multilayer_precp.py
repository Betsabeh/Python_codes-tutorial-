# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:34:06 2022

@author: betsa
"""
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import matplotlib as plt
# 1-create iris dataset
data, info=tfds.load('iris', with_info=True)
###print(info)
# 2-test and train datasets
tf.random.set_seed(1)
data_tr=data['train']
New_data=data_tr.shuffle(150, reshuffle_each_iteration=False)
Data_train=New_data.take(100)
Data_test=New_data.skip(100)
# 3- convert dictionary to tuple
ds_train=Data_train.map(lambda x: (x['features'],x['label']))
ds_test=Data_test.map(lambda x:   (x['features'],x['label']))
# 4-create model
mdl=tf.keras.Sequential([tf.keras.layers.Dense(16, activation='sigmoid',name='fc1', input_shape=(4,)),tf.keras.layers.Dense(3, activation='softmax', name='fc2')])
mdl.summary()
mdl.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 5-Train model
num_epoch=100
batch_size=2
training_size=100
step_per_epoch=np.ceil(training_size/batch_size)
ds_train=ds_train.shuffle(buffer_size=training_size)
ds_train=ds_train.repeat()
ds_train=ds_train.batch(batch_size=batch_size)
hist=mdl.fit(ds_train, epochs=num_epoch, steps_per_epoch=step_per_epoch, verbose=0)
# 6-visualize the training
temp=hist.history
fig=plt.pyplot.figure(figsize=(10,5))
ax=fig.add_subplot(1,2,1)
ax.plot(temp['loss'],lw=3)
ax.set_title('training loss')
ax=fig.add_subplot(1,2,2)
ax.plot(temp['accuracy'])
ax.set_title('accuracy')
# 7-test and validation
results=mdl.evaluate(ds_test.batch(50),verbose=0)
print('Test loss={:.4f} ,  Test Accuracy={:.4f}'.format(*results))
