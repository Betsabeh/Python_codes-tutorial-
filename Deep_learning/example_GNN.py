# this file contains codes for simple GNN network
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
# step1-define function for generator network
def make_generator(num_h_layer=1,num_h_units=100,num_output_units=784):
  mdl=tf.keras.Sequential()
  for i in range(num_h_layer):
    mdl.add(tf.keras.layers.Dense(units=num_h_units,use_bias=False))
    mdl.add(tf.keras.layers.LeakyReLU())
    mdl.add(tf.keras.layers.Dense(units=num_output_units,activation='tanh'))
  return mdl
# step 2-define function for Descriminator
def make_discriminator(num_h_layers=1,num_h_units=100,num_output_units=1):
  mdl=tf.keras.Sequential()
  for i in range(num_h_layers):
    mdl.add(tf.keras.layers.Dense(units=num_h_units))
    mdl.add(tf.keras.layers.LeakyReLU())
    mdl.add(tf.keras.layers.Dropout(rate=0.5))
  mdl.add(tf.keras.layers.Dense(units=num_output_units,activation=None))
  return mdl

# step 3-build model
image_size=(28,28)
z_size=20
gen_num_h_layers=1
gen_num_h_units=100
dis_num_h_layers=1
dis_num_h_units=100
tf.random.set_seed(1)
gen_mdl=make_generator(num_h_layer=gen_num_h_layers, 
                       num_h_units=gen_num_h_units, 
                       num_output_units=np.prod(image_size))
gen_mdl.build(input_shape=(None,z_size))
gen_mdl.summary()

dis_mdl=make_discriminator(num_h_layers=dis_num_h_layers, 
                           num_h_units=dis_num_h_units, 
                           num_output_units=1)
dis_mdl.build(input_shape=(None,np.prod(image_size)))
dis_mdl.summary()

# step4-read dataset and preprocessing
mnist_b=tfds.builder('mnist')
mnist_b.download_and_prepare()
mnist_data=mnist_b.as_dataset(shuffle_files=False)

def preprocessing(ex,mode='uniform'):
  image=ex['image']
  image=tf.image.convert_image_dtype(image,tf.float32)
  image=tf.reshape(image,[-1])
  image=image*2-1.0
  if mode=='uniform':
    input_z=tf.random.uniform(shape=(z_size,),minval=-1.0,maxval=1.0)
  elif mode=='normal':
    input_z=tf.random.normal(shape=(z_size,))
  return input_z, image

mnist_train=mnist_data['train']
mnist_train=mnist_train.map(preprocessing)
mnist_train=mnist_train.batch(32,drop_remainder=True)
input_z,input_real=next(iter(mnist_train))
print('input z shape:', input_z.shape)
# g-out=fake image produce by gen net
g_out=gen_mdl(input_z)
print('output gen shape:',g_out.shape)
#use g_out to fed dis net and produce d_logit_fake
d_logit_fake=dis_mdl(g_out)
# fed the dis network with real images and produce d_logit_real
d_logit_real=dis_mdl(input_real)

#  step5- training GAN
loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
# loss for gen
g_label_real=tf.ones_like(d_logit_fake)
g_loss=loss_fn(y_true=g_label_real,y_pred=d_logit_fake)
print('loss gen net:',g_loss)
# loss for des
d_label_real=tf.ones_like(d_logit_real)
d_label_fake=tf.zeros_like(d_logit_fake)
d_loss_fake=loss_fn(y_true=d_label_fake,y_pred=d_logit_fake)
d_loss_real=loss_fn(y_true=d_label_real,y_pred=d_logit_real)
print('loss dis net real:',d_loss_real,'\n loss dis net fake:',d_loss_fake)


