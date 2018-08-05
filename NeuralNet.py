#含有一个隐层的神经网络识别mnist
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

MAX_STEP = 3001
batchSize = 100
mnist = input_data.read_data_sets("F:\MNIST_data",one_hot=True)
#定义输入层
with tf.name_scope("input"):
    images = tf.placeholder(tf.float32,[None,784],name = 'x_input')
    labels = tf.placeholder(tf.float32,[None,10],name = "y_input")
with tf.name_scope("input_reshape"):
    images_reshape = tf.reshape(images,shape=[-1,28,28,1])
#定义300个节点的隐层
with tf.name_scope("hidden_layer"):
    w1 = tf.Variable(tf.truncated_normal([784,300],mean=0.0,stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([300],stddev = 0.1))
    hideLayer = tf.nn.relu(tf.matmul(images,w1)+b1)

#定义输出层
with tf.variable_scope("output_layter"):
    w2 = tf.Variable(tf.truncated_normal([300,10],mean=0.0,stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([10],stddev = 0.1))
    outputLayer = tf.nn.softmax(tf.matmul(hideLayer,w2)+b2)

#定义交叉熵损失函数
loss = tf.reduce_mean(tf.reduce_sum(-labels*tf.log(outputLayer),axis= 1))
train = tf.train.AdagradOptimizer(0.3).minimize(loss)

merge = tf.summary.merge_all()
#定义会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_time = 0
    train_writer = tf.summary.FileWriter("C:/train", sess.graph)
    for step in range( MAX_STEP):
        x,y = mnist.train.next_batch(batchSize)
        start = time.time()
        sess.run([train],feed_dict={images:x,labels:y})
        end =time.time()
        total_time+=(end- start)
        if step%100==0:
            bAcc = tf.equal(tf.argmax(labels,axis=1),tf.argmax(outputLayer,axis=1))
            fAcc = tf.cast(bAcc, tf.float32)
            acc = tf.reduce_mean(fAcc)
            acc = sess.run(acc, feed_dict={images: mnist.test.images, labels: mnist.test.labels})
            print("第%d次迭代:准确率:%%%0.3f" % (step, acc * 100))
    avg_time = total_time * 1.0 / MAX_STEP
    print("训练总耗时:%0.3f s，平均反向传播一次耗时:%0.3f s" % (total_time, avg_time))
