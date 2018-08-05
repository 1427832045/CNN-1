#使用了1*1卷积
#使用两个3*3卷积核代替5*5
import tensorflow as tf
import time
import datetime
import math
#定义卷基层
def conv2d(input,name,out_num,weight_height,weight_width,stride_height,stride_width):
    global parameters
    with tf.name_scope(name):
        in_num = input.get_shape()[-1]#返回输入的节点数
        #xavier初始化方法
        weight = tf.get_variable(name+"weight",shape = [weight_height,weight_width,in_num,out_num],\
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.zeros(shape = [out_num]),name = name+"bias")
        parameters += [weight,bias]
        conv = tf.nn.conv2d(input,weight,strides=[1,stride_height,stride_width,1],padding='SAME')
        z = tf.nn.bias_add(conv,bias)
        relu = tf.nn.relu(z)
        return relu
#定义全连接
def fc(input,name,out_num):
    in_num = input.get_shape()[-1]#输入的input认为是二维tensor[batchsize,in_num]
    with tf.name_scope(name):
        weight = tf.get_variable(name+"weight",shape=[in_num,out_num],dtype=tf.float32,\
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name+"bias",initializer=tf.constant(0.1,shape=[out_num]))
        relu = tf.nn.relu_layer(input,weight,bias)
        return relu
#定义最大池化层
def max_pool(input,name,weight_height,weight_width,stride_height,stride_width):
    pool = tf.nn.max_pool(input,ksize=[1,weight_height,weight_width,1],\
                strides=[1,stride_height,stride_width,1],padding='SAME',name = name)#注意ksize四维的含义
    return pool

def inference(images):
    #第一段卷积
    conv1_1 = conv2d(images,"conv1_1",64,3,3,1,1)
    conv1_2 = conv2d(conv1_1,"conv1_2",64,3,3,1,1)
    pool1 = max_pool(conv1_2,"pool1",2,2,2,2)

    #第二段卷积
    conv2_1 = conv2d(pool1,"conv2_1",128,3,3,1,1)
    conv2_2 = conv2d(conv2_1,"conv2_2",128,3,3,1,1)
    pool2 = max_pool(conv2_2,"pool2",2,2,2,2)

    #第三段卷积
    conv3_1 = conv2d(pool2,"conv3_1",256,3,3,1,1)
    conv3_2 = conv2d(conv3_1, "conv3_2", 256, 3, 3, 1, 1)
    conv3_3 = conv2d(conv3_2,"conv3_3",256,3,3,1,1)
    pool3 = max_pool(conv3_3,"pool3",2,2,2,2)

    #第四段卷积
    conv4_1 = conv2d(pool3, "conv4_1", 512, 3, 3, 1, 1)
    conv4_2 = conv2d(conv4_1, "conv4_2", 512, 3, 3, 1, 1)
    conv4_3 = conv2d(conv4_2, "conv4_3", 512, 3, 3, 1, 1)
    pool4 = max_pool(conv4_3, "pool4", 2, 2, 2, 2)

    #第五段卷积
    conv5_1 = conv2d(pool4, "conv5_1", 512, 3, 3, 1, 1)
    conv5_2 = conv2d(conv5_1, "conv5_2", 512, 3, 3, 1, 1)
    conv5_3 = conv2d(conv5_2, "conv5_3", 512, 3, 3, 1, 1)
    pool5 = max_pool(conv5_3, "pool5", 2, 2, 2, 2)

    #将数据转为二维[bathc_size,in_num]
    print(pool5.shape)
    in_num = pool5.get_shape()[1].value*pool5.get_shape()[2].value*pool5.get_shape()[3].value
    pool5_flat = tf.reshape(pool5,shape=[-1,in_num])

    #三层全连接,每层后面加droupout
    fc1 = fc(pool5_flat,"fc1",4096)
    fc1_drop = tf.nn.dropout(fc1,keep_prob,name="fc1_drop")


    fc2 = fc(fc1,"fc2",4096)
    fc2_drop = tf.nn.dropout(fc2,keep_prob,name='fc2_drop')

    softmax  = tf.nn.softmax(fc(fc2,'softmax',1000))
    predict = tf.argmax(softmax,1)
    l2_loss = tf.nn.l2_loss(fc2,"l2_loss")
    grad = tf.gradients(l2_loss,parameters)
    return predict,grad

def cal_run_time(sess,tensor,feed,info_string):
    iter_num = 10
    total_time = 0
    total_time_square = 0
    ave = 0
    for i in range(10 + iter_num):
        start = time.time()
        sess.run(tensor)  # 执行tensor
        end = time.time()  # 返回时间戳，单位为秒
        duration = end - start
        if i >= 10:
            print("%s:%s,第%d次循环,耗时 = %.3f s" % (datetime.datetime.now(), info_string, i - 10, duration))
            total_time += duration
            total_time_square += duration * duration
    mean_time = total_time / iter_num  # 计算平均时间
    variance = total_time_square / iter_num - mean_time * mean_time
    std_dev = math.sqrt(variance)
    print("%s:%s,总循环次数 %d,平均用时 = %f s,标准差 = %f" % (datetime.datetime.now(), info_string, iter_num, mean_time, std_dev))

if __name__ == "__main__":
    batch_size = 32
    image_height = 224
    image_width = 224
    parameters = []
    keep_prob = tf.placeholder(tf.float32)#训练时小于，用来防止过拟合；预测时=1，利用全部特征
    images = tf.Variable(initial_value=tf.truncated_normal\
                         (shape=[batch_size,image_height,image_width,3]))#生成随机数据
    predict,grad = inference(images)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    cal_run_time(sess,predict,{keep_prob:1},"前向运算")
    cal_run_time(sess,grad,{keep_prob:0.5},"反向传播")
