# Welcome to the journey to tensflow with me.
import tensorflow as tf

# Before creating your network, you need to download the training and testing data from a well_known database -- MNIST
# input_data has two parts : training set of 6000 rows and testing set of 1000 rows.
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

# Each element has two components -- image and label
# An image has 28*28 pixels. We may as well regard it as a 784 long vector, which ignores its complex struction.
# Thus mnist.train.images is a [6000,784] tensor. mnist.train.label is a [6000,10] matrix. (Each image depicts a number
# 0 to 9)

# Describe a placeholder in your tensorflow graph. Keep it in your mind : when you are coding in tensorflow, you are
# drawing a "tensor flow"(just like flow chart). A placeholder requires your input when running.
x=tf.placeholder("float",[None,784]) # "None" means infinit and arbitrary.

# Using softmax
# Declaring the weight and parameters.
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

# Convolution = c = xW+b; y = normalized [exp(c)]
# Declaring a op of convolution.
y=tf.nn.softmax(tf.matmul(x,W)+b)

# Adding another placeholder to input correct answer(label).
y_=tf.placeholder("float",[None,10])

# Using cross entropy as cost function
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

# TensorFlow automatically uses backpropogation algorithm to minimize the cost function. Alpha=0.01.
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Adding an op to initialize the variables. !!! You may get a warning if take tf.initialize_all_variables() instead.
init=tf.global_variables_initializer();

# Start our graph
sess=tf.Session()
sess.run(init)

# Train 1000 times. Each time select an arbitrary set for training.
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# Assessing the property of our model.
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) # Return a bool vector.
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
