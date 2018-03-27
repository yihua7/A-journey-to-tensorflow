import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

# Creat two placeholders.
x=tf.placeholder("float",[None,784])
y_=tf.placeholder("float",[None,10])

# Break the symmetry of W.
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# Prevent bias keeping equaling to 0.
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# Convolution. Setting stride = 1, padding = 0. Ensure the sizes of input and output are the same.
# Input has four dimensions : [batch, height, width, channels], and youu can set the stride of your filter in these dimensions.
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# Use 2*2 model to max pool.
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# The first layer.
# Convolution : the size of patch is [5,5]; 1 input channel and 32 output channel.
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

# Transform x into a 4-dimensional vector. 1st -1 means "adjust as necessary to match the size needed for the full tensor."
# 2th and 3th dimensions correspond to the height and width of images. 4th d stands for color channel.
x_image=tf.reshape(x,[-1,28,28,1])


# Output of 1st layer. Use ReLU function as activation function.
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

# The second layer.
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

# Full connected layer.
W_fcl=weight_variable([7*7*64,1024])
b_fcl=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fcl=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)

keep_prob=tf.placeholder("float")
h_fcl_drop=tf.nn.dropout(h_fcl,keep_prob)

# Output layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fcl_drop,W_fc2)+b_fc2)
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
#sess=tf.Session(tf.ConfigProto(gpu_options=gpu_options))
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(session=sess,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g" %(i,train_accuracy))
    train_step.run(session=sess,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g" %accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_:mnist.test.lebels,keep_prob:1.0}))
