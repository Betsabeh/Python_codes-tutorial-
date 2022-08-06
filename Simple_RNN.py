import tensorflow as tf
import pandas as pd
import numpy as np
#RNN model
Rnn=tf.keras.layers.SimpleRNN(units=1,use_bias=True,return_sequences=True)
Rnn.build(input_shape=(None,None,5))
Rnn.compute_output_shape(input_shape=(None,None,5))
W_xh,W_oo,W_bh=Rnn.weights
print('----Initial Weights------')
print('\ninput-hidden-weights:',W_xh.numpy())
print('\n output-output-weights:',W_oo.numpy())
print('\n bias-hidden-weights:',W_bh.numpy())
# compute manually
seq=tf.convert_to_tensor([[1]*5,[3]*5,[2]*5],dtype=tf.float32)
print('---------------Seq-----------------')
print(seq)
output=Rnn(tf.reshape(seq,shape=(1,3,5)))
print('----------------------------------')
out_man=[]
for t in range(len(seq)):
  xt=tf.reshape(seq[t],(1,5))
  print('time Step', t,'=>')
  print('Input:',xt.numpy())
  ht=tf.matmul(xt,W_xh)+W_bh
  print('Hidden:',ht.numpy())
  if t>0:
    Prev_output=out_man[t-1]
  else:
    Prev_output=tf.zeros(shape=(ht.shape))
  ot=ht+tf.matmul(Prev_output,W_oo)
  print('Output node:',ot.numpy())
  ot=tf.math.tanh(ot)
  out_man.append(ot)
  print('Output Manual:',ot.numpy())
  print('output RNN:',output[0][t].numpy()) 
  print('----------------------------------')  
