import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import datetime
data = input_data.read_data_sets('F://MNIST_data/',one_hot=True)
print("load over")
nbatchSize = 1000
images = tf.placeholder(tf.float32,shape = [None,784])
labels = tf.placeholder(tf.float32,shape = [None,10])
w = tf.Variable(tf.truncated_normal(shape = [784,10]))
b = tf.Variable(tf.truncated_normal(shape=[10]))
wx_plus_b = tf.matmul(images,w)+b
print(wx_plus_b.get_shape())
y = tf.nn.softmax(wx_plus_b)#axis =none,默认对最后一位做softmax
loss = tf.reduce_mean(tf.reduce_sum(-labels*tf.log(y),axis=1))#定义交叉熵损失函数,注意axis=1表示按照行求和，=0表示按照列求和
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)#使用梯度下降法优化

# arr1 = tf.Variable([[1,2],[4,5]])
# arr2 = tf.Variable([[1,1],[1,1]])
# arr3 = tf.multiply(arr1,arr2)
# arr4 = tf.matmul(arr1,arr2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(arr3))
    # print(sess.run(arr4))
    #batch_x, batch_y = data.train.next_batch(nbatchSize)  # 每次随机选取nbatchSize个大小的样本进行训练。
    total_time = 0
    for i in range(1000):
        start = time.time()
        #print(sess.run(b))
        batch_x,batch_y = data.train.next_batch(nbatchSize)#每次随机选取nbatchSize个大小的样本进行训练。
        sess.run(train,feed_dict={images:batch_x,labels:batch_y})
        end = time.time()
        total_time +=(end - start)
        if i%100 == 0:
            res = tf.equal(tf.argmax(labels,axis=1),tf.argmax(y,axis=1))#axis = 1按行求最大值对应的索引
            acc = tf.reduce_mean(tf.cast(res,tf.float32))#将bool转为float类型
            fAcc = sess.run(acc,feed_dict = {images:data.test.images,labels:data.test.labels})#在使用t.eval()时，等价于：tf.get_default_session().run(t).
            print("第%d次迭代:准确率:%%%0.3f"%(i,fAcc*100))
    avg_time = total_time*1.0/i
    print("训练总耗时:%0.3f s，平均反向传播一次耗时:%0.3f s"%(total_time,avg_time))