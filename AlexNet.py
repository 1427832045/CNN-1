#2012ILSVRC冠军:5个卷积层+3个全连接，前两个卷基层后面加上lrn，主要特点：
#1.使用了relu激活函数，避免了梯度消失的情况
#2.使用了dropout来关闭部分神经元，避免过拟合
#3.使用了lrn，提高了百分之一的准确率
#4.使用了最大池化层，比较均值池化，避免了模糊特征，同时使核大小大于步长，增加了特征的重叠，提高了泛化能力
#5.使用了cuda加速q

#本程序用来生成模拟数据用来测试alexnet前向和后向运算的时间
import time
import datetime
import math
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
parameters = []

def print_activation(tensor):#打印出tensor名称和尺寸
    print(tensor.op.name,'',tensor.get_shape().as_list())

def inference(image):
    global parameters
    #第一个卷积层
    with tf.variable_scope("conv1") as scope:
        w = tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.1),name = "w")
        print_activation(w)
        b = tf.Variable(tf.truncated_normal([96]))
        conv1 = tf.nn.relu(tf.nn.conv2d(image,w,strides=[1,4,4,1],padding='SAME')+b)
        print_activation(conv1)
        parameters+=[w, b]
        #lrn1 = tf.nn.lrn(input=conv1, depth_radius=5, bias=1, alpha=0.001/9, beta=0.75)
        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
        print_activation(pool1)

    #第二个卷积层
    with tf.name_scope("conv2") as scope:
        w = tf.Variable(tf.truncated_normal([5,5,96,192],stddev=0.1))
        b = tf.Variable(tf.truncated_normal([192]))
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1,w,strides=[1,1,1,1],padding='SAME')+b)
        print_activation(conv2)
        #lrn2 = tf.nn.lrn(input=conv2, depth_radius=5, bias=1, alpha=0.001 / 9, beta=0.75)
        pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
        print_activation(pool2)
        parameters+=[w,b]
    #第三个卷积层,没有池化层
    with tf.name_scope("conv3") as scope:
        w = tf.Variable(tf.truncated_normal([3,3,192,384],stddev=0.1))
        b = tf.Variable(tf.truncated_normal([384]))
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2,w,strides=[1,1,1,1],padding='SAME')+b)
        parameters+=[w,b]
        print_activation(conv3)
    #第四个卷积层
    with tf.name_scope("conv4") as scope:
        w = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1))
        b = tf.Variable(tf.truncated_normal([256]))
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3,w,strides=[1,1,1,1],padding='SAME')+b)
        parameters+=[w,b]
        print_activation(conv4)
    #第五个卷积层
    with tf.name_scope("conv5") as scope:
        w = tf.Variable(tf.truncated_normal([3,3,256,256],stddev=0.1))
        b = tf.Variable(tf.truncated_normal([256]))
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4,w,strides=[1,1,1,1],padding='SAME')+b)
        print_activation(conv5)
        parameters += [w, b]
        pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
        print_activation(pool5)

    return (pool5,parameters)

def cal_run_time(sess,tensor,info_string):#统计tensor耗时
    iter_num = 10
    total_time = 0
    total_time_square = 0
    ave = 0
    for i in range(10+iter_num):
        start = time.time()
        sess.run(tensor)#执行tensor
        end = time.time()#返回时间戳，单位为秒
        duration = end - start
        if i>= 10:
            print("%s:%s,第%d次循环,耗时 = %.3f s"%(datetime.datetime.now(),info_string,i-10,duration))
            total_time += duration
            total_time_square += duration*duration
    mean_time = total_time/iter_num#计算平均时间
    variance = total_time_square/iter_num - mean_time*mean_time
    std_dev = math.sqrt(variance)
    print("%s:%s,总循环次数 %d,平均用时 = %f s,标准差 = %f"%(datetime.datetime.now(),info_string,iter_num,mean_time,std_dev))


if __name__ == "__main__":
    nBatchSize = 32
    images = tf.Variable(tf.random_normal(shape=[nBatchSize,224,224,3]))#生成模拟数据用来测试
    pool5,parameters = inference(images)


    sess = tf.InteractiveSession()#定义会话
    sess.run(tf.global_variables_initializer())#变量初始化

    # 统计一次前向运算的时间
    cal_run_time(sess,pool5,"前向运算")
    #print(sess.run(parameters))
    #统计一次反向计算梯度的时间
    loss = tf.nn.l2_loss(pool5)
    grad = tf.gradients(loss,parameters)
    cal_run_time(sess,grad,"反向梯度")


    # for i in range(step):
    #     x,y = mnist.train.next_batch(nBatchSize)
    #     image = tf.reshape(x,shape=[nBatchSize,28,28,1])
    #     sess.run(inference(image))
