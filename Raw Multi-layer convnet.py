from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x=tf.placeholder("float",[None,784])
y_=tf.placeholder("float",[None,10])

x_image=tf.reshape(x,[-1,28,28,1])
W1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[32]))

out1_=tf.nn.relu(tf.nn.conv2d(x_image,W1,strides=[1,1,1,1],padding='SAME')+b1)
out1=tf.nn.max_pool(out1_,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[64]))

out2_=tf.nn.relu(tf.nn.conv2d(out1,W2,strides=[1,1,1,1],padding='SAME')+b2)
out2=tf.nn.max_pool(out2_,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
out2_r=tf.reshape(out2,[-1,7*7*64])

W3=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b3=tf.Variable(tf.constant(0.1,shape=[1024]))

out3=tf.nn.relu(tf.matmul(out2_r,W3)+b3)
keep_prob=tf.placeholder("float")
out3_d=tf.nn.dropout(out3,keep_prob=keep_prob)

W4=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b4=tf.Variable(tf.constant(0.1,shape=[10]))

out4=tf.nn.relu(tf.matmul(out3_d,W4)+b4)
y=tf.nn.softmax(out4)

cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(prediction,"float"))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    x_batch,y_batch=mnist.train.next_batch(50)
    if i%100==0:
        print("%d train's accuracy is %g" %(i,accuracy.eval(session=sess,feed_dict={x:x_batch,y_:y_batch,keep_prob:1})))
    sess.run(train,feed_dict={x:x_batch,y_:y_batch,keep_prob:0.5})

print("The final accuracy is %g",accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))