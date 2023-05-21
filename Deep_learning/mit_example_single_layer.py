class my_single(tf.keras.layers.Layer):
  def __init__(self,num_output):
    super(my_single,self).__init__()
    self.outputnodes=num_output
  
  def build(self,input_shape):
    d=input_shape[-1] # samples dimension
    self.W=self.add_weight("weight", shape=[d,self.outputnodes])
    self.b=self.add_weight("bias",shape=[1,self.outputnodes])
  
  def cal(self,x):
    Z=tf.matmul(x,self.W)
    Z=tf.add(Z,self.b)
    out=tf.sigmoid(Z)
    return out

layer=my_single(3)
layer.build((1,2))
X_input=tf.constant([[1,2.0]],shape=(1,2))
y=layer.cal(X_input)
print('output={}'.format(y.numpy()))
