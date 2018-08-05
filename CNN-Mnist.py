import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

def weight_variable(shape):
    weight = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return weight
def bias_variable(shape):
    bias = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return bias

MAX_STEP = 3001
nBatchSize = 100
mnist = input_data.read_data_sets("F://MNIST_data",one_hot=True)
print(mnist.test.images.shape)
images = tf.placeholder(shape=[None,784],dtype=tf.float32)#从mnist中拿到的图像数据是1维的
matImages = tf.reshape(images,shape=[-1,28,28,1])#将一维数据转为28*28的矩阵数据，[样本数量，图像高，图像宽，颜色通道数]
labels = tf.placeholder(tf.float32,shape=[None,10])

#定义第一个卷积层
w1 = weight_variable(shape = [5,5,1,32])#注意这里1表示颜色通道
b1 = bias_variable(shape=[32])
conv1 = tf.nn.relu(tf.nn.conv2d(matImages,w1,strides=[1,1,1,1],padding="SAME")+b1)
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#定义第二个卷积层
w2 = weight_variable(shape = [5,5,32,64])#注意第三个参数，应该为上一层的节点数
b2 = bias_variable(shape=[64])
conv2 = tf.nn.relu(tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding="SAME")+b2)
pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#定义一个全连接
pool2_flat = tf.reshape(pool2,shape=[-1,7*7*64])
w3 = weight_variable(shape=[7*7*64,1024])
b3 = bias_variable(shape=[1024])
fc1 = tf.nn.relu(tf.matmul(pool2_flat,w3)+b3)

#定义softmax
w4 = weight_variable(shape=[1024,10])
b4 = bias_variable(shape =[10])
predict = tf.nn.softmax(tf.matmul(fc1,w4)+b4)

#定义优化方法
cross_entropy = tf.reduce_mean(tf.reduce_sum(-labels*tf.log(predict),axis=1))#定义交叉熵损失函数
train = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)#选择优化方法

#计算准确率
bAcc = tf.equal(tf.argmax(labels, axis=1), tf.argmax(predict, axis=1))  # 将预测结果与标记值对比
fAcc = tf.cast(bAcc, tf.float32)  # 将bool类型转为float32
acc = tf.reduce_mean(fAcc)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化所有变量
    total_time = 0
    for step in range(MAX_STEP):
        batch_x,batch_y = mnist.train.next_batch(nBatchSize)
        #print(sess.run(acc,feed_dict={images:batch_x,labels:batch_y}))
        start = time.time()
        sess.run(train,feed_dict={images:batch_x,labels:batch_y})
        end = time.time()
        total_time += (end - start)
        if step%100==0:
        # print(sess.run(bAcc),feed_dict={images:batch_x,labels:batch_y})
        #if step%50==0:
            accuracy = sess.run(acc,feed_dict={images:mnist.test.images,labels:mnist.test.labels})
            print("第%d次迭代:准确率:%%%0.3f" % (step, accuracy * 100))
    avg_time = total_time * 1.0 / MAX_STEP
    print("训练总耗时:%0.3f s，平均反向传播一次耗时:%0.3f s" % (total_time, avg_time))

