import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

def get_weight(shape,lam):
    # Declare a variable as weight.
    var=tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32)
    # Add the regularization loss to the collection 'losses'.
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lam)(var))
    return var

x=tf.placeholder(tf.float32,shape=(None,784))
x_image=tf.reshape(x,[-1,28,28,1])
y_=tf.placeholder(tf.float32,shape=(None,10))
batch_size=8

# First layer of convnet.
cur_layer=x_image

dimension=[1,32,64]

# Using a loop to create neutral network.
for i in range(1,2):
    weight=get_weight([dimension[i-1],5,5,dimension[i]],0.001)
    bias=tf.Variable(tf.constant(dimension[i]))
    cur_layer=tf.nn.relu(tf.nn.max_pool(tf.nn.conv2d(cur_layer,weight,strides=[1,1,1,1],padding='SAME')+bias,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

cur_layer=tf.reshape(cur_layer,[None,7*7*64])
di=[7*7*64,1024,10]

for i in range(1,2):
    weight=get_weight([di[i-1],di[i]],0.001)
    bias=tf.Variable(tf.constant(di[i]))
    cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)

y=tf.nn.softmax(cur_layer)

entropy=-tf.reduce_sum(y_*tf.log(y))
tf.add_to_collection('losses',entropy)
loss=tf.add_n(tf.get_collection('losses'))

train=tf.train.AdamOptimizer(1e-4).minimize(loss)

prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(prediction,"float"))

gpu_options=tf.GPUOptions(per_process_memory_fraction=0.33)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(50)
    if i%100==0:
        print("The %dth training has accuracy %g" %(i,accuracy.eval(session=sess,feed_dict={x:batch_xs,y_:batch_ys})))
    train.run(session=sess,feed_dict={x:batch_xs,y:batch_ys})

print("Final accuracy is %g" %accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))