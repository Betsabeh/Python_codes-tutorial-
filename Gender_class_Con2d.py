import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------------------
def preprocessing(Sample,size=(64,64)):
  image=Sample['image']
  label=Sample['attributes']['Male']
  # crop 
  #Cr_image=tf.image.central_crop(image,0.8)
  Cr_image=tf.image.crop_to_bounding_box(image,offset_height=15,offset_width=0,
                                         target_width=178,target_height=178)
  # resize
  Re_image=tf.image.resize(image,size=size)
  #fig=plt.figure(figsize=(12,6))
  #ax=fig.add_subplot(1,2,1)
  #ax.imshow(image)
  #ax=fig.add_subplot(1,2,2)
  #ax.imshow(Cr_image)
  return Re_image/255.0  ,  tf.cast(label, tf.int32)
#-------------------------------------
def build_CNN(num_filter):
  mdl=tf.keras.Sequential()
  #first conv layer
  mdl.add(tf.keras.layers.Conv2D(filters=num_filter,kernel_size=(3,3),padding='same',
                                 activation='relu'))
  mdl.add(tf.keras.layers.MaxPool2D(strides=(2,2),pool_size=(2,2)))
  mdl.add(tf.keras.layers.Dropout(0.5))
  #second conv layer
  mdl.add(tf.keras.layers.Conv2D(filters=num_filter*2,kernel_size=(3,3),padding='same',
                                 activation='relu'))
  mdl.add(tf.keras.layers.MaxPool2D(strides=(2,2),pool_size=(2,2)))
  mdl.add(tf.keras.layers.Dropout(0.5))
  # third conv layer
  mdl.add(tf.keras.layers.Conv2D(filters=4*num_filter,kernel_size=(3,3),padding='same',
                                 activation='relu'))
  mdl.add(tf.keras.layers.MaxPool2D(strides=(2,2),pool_size=(2,2)))
  # fourth conv layer
  mdl.add(tf.keras.layers.Conv2D(filters=8*num_filter,kernel_size=(3,3),padding='same',
                                 activation='relu'))
  mdl.add(tf.keras.layers.GlobalAveragePooling2D())
  # Output
  mdl.add(tf.keras.layers.Dense(units=1,activation=None))
  mdl.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
  mdl.build(input_shape=(None,64,64,3))
  mdl.summary()
  return mdl
#-------------------------------------
#1-load dataset
data=tfds.load('CelebA')
Train_data=data['train']
Test_data=data['test']
Valid_data=data['validation']
#-------------------------------------
#2-Select part of dataset
Train_data=Train_data.take(16000)
Valid_data=Valid_data.take(1000)
#-------------------------------------
#Parameters
img_size=(64,64)
num_filter=32
epochs=60
batch_size=64
#-------------------------------------
#3-preprocessing
Train_Ds=Train_data.map(lambda sample:preprocessing(sample,size=img_size))
Valid_Ds=Valid_data.map(lambda sample: preprocessing(sample,size=img_size))
Test_Ds=Test_data.map(lambda sample:preprocessing(sample,size=img_size))
Train_Ds=Train_Ds.batch(batch_size=batch_size)
Valid_Ds=Valid_Ds.batch(batch_size=batch_size)
#------------------------------------
#4-create Conv2D model
mdl=build_CNN(num_filter)
#------------------------------------
#5-Train model
mdl_hist=mdl.fit(Train_Ds,epochs=epochs,verbose=1, validation_data=Valid_Ds)
hist=mdl_hist.history
x=range(epochs)
fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(1,2,1)
ax.plot(x,hist['loss'],'-o',label='Train Loss')
ax.plot(x,hist['val_loss'],'--*',label='validation Loss')
ax.legend(fontsize=10)
ax.set_xlabel('epochs')
ax.set_ylabel('Loss')
ax=fig.add_subplot(1,2,2)
ax.plot(x,hist['accuracy'],'-o',label='Train accuracy')
ax.plot(x,hist['val_accuracy'],'--*',label='Validation accuracy')
ax.legend(fontsize=10)
ax.set_xlabel('epochs')
ax.set_ylabel('Accuracy')
plt.show()
#--------------------------------------
#6-Test
# Test Batch
T1=Test_Ds.batch(batch_size=batch_size)
Test_result=mdl.evaluate(T1)
print('-------------Test Bacth result-----------------')
print('Test Accuracy:',Test_result[1])
# Predict Probabilty
T2=Test_Ds.batch(batch_size=len(Test_Ds))
Pred_logit=mdl.predict(T2)
Prob=tf.sigmoid(Pred_logit)


