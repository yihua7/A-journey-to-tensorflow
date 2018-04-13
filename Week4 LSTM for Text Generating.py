# Practice of Long-Short-Term-Memory
# Recurrent Nueral Networks for Language Modeling

import time
import numpy as np
import tensorflow as tf

# Loading data
with open('C:/Users/Dell/Documents/GitHub/Data/anna.txt','r') as f:
    text=f.read()

# Creating a char set of data
vocab=set(text)
# Char-Num Dictionary
vocab_to_int={c:i for i ,c in enumerate(vocab)}
# Num-Char Dictionary
int_to_vocab=dict(enumerate(vocab))

# Transcoding text from char to num
encoded=np.array([vocab_to_int[c] for c in text],dtype=np.int32)

def get_batches(arr, n_seqs, n_steps):
    """
    Cutting array in to mini batches
    :param arr: array
    :param n_seqs: size of a batch
    :param n_steps:  size of a sequence
    """
    batch_size=n_seqs * n_steps
    n_batches=int(len(arr) / batch_size)

    arra=arr[:batch_size * n_baches]
    arr=arr.reshape((n_seqs,-1))

    for n in range(0, arr.shape[1], n_steps):
        x=arr[: n: n + n_steps]
        y=np.zeros_like(x)
        y[:,:-1],y[:,-1]=x[:,1:],y[:,0]
        yield x,y

def build_inputs(num_seqs, num_steps):
    inputs=tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets=tf.placeholder(tf.int32, shape=(num_seqs, num_steps),name='targets')
    keep_prob=tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell=tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    initial_state=cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    seq_output=tf.concat(lstm_output, 1)
    x=tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w=tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b=tf.Variable(tf.zeros(out_size))

    logits=tf.matmul(x,softmax_w)+softmax_b
    out=tf.nn.softmax(logits, name='predictions')

    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot=tf.one_hot(targets, num_classes)
    y_reshaped=tf.reshape(y_one_hot, logits.getshape())

    loss=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss=tf.reduce_mean(loss)

    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    tvars=tf.trainable_variables()
    grads, _=tf.clip_by_global_norm(tf.gradients(loss,tvars), grad_clip)
    train_op=tf.train.AdamOptimizer(learning_rate)
    optimizer=train_op.apply_gradients(zip(grads, tvars))
    return optimizer



class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):

        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


batch_size = 100  # Sequences per batch
num_steps = 100  # Number of sequence steps per batch
lstm_size = 512  # Size of hidden layers in LSTMs
num_layers = 2  # Number of LSTM layers
learning_rate = 0.001  # Learning rate
keep_prob = 0.5  # Dropout keep probability

# In[16]:

epochs = 20
# 每n轮进行一次变量保存
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            # control the print lines
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e + 1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[19]:

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    生成新文本

    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int_to_vocab[c])

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)


# Here, pass in the path to a checkpoint and sample from the network.

# In[20]:

tf.train.latest_checkpoint('checkpoints')

# In[26]:

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)

# In[22]:

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

# In[23]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

# In[24]:

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)