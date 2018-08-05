import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import datetime
import math
trunc_normal = lambda stddev:tf.truncated_normal_initializer(0.1,stddev)

def inception_v3_arg_scope(weight_decay = 0.0004,stddev = 0.1,batch_norm_var_collectin='moving_vars'):
    batch_norm_params = {'decay':0.9997,'epsilon':0.001}

def inception_v3_base(input,scope = None):
    #定义卷积层
    with tf.variable_scope(scope,"InceprionV3"):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
            net = slim.conv2d(input,num_outputs=32,kernel_size=[3,3],stride=2)#输出尺寸(299-3)/2+1=149
            net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3])
            net = slim.conv2d(net,64,[3,3],padding="SAME")#补一圈0
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2)
            net = slim.conv2d(net,80,[1,1])
            net = slim.conv2d(net,192,[3,3])
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2)#输出为35*35
        #定义第一个模块组，包含3个模块
        # 第一个模块
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d], stride=1, padding="SAME"):
            with tf.variable_scope("Module_1-1"):
                with tf.variable_scope("branch_1"):
                    branch_1 =slim.conv2d(net,64,[1,1])
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,48,[1,1])
                    branch_2 = slim.conv2d(branch_2,64,[5,5])
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.conv2d(net,64,[1,1])
                    branch_3 = slim.conv2d(branch_3,96,[3,3])
                    branch_3 = slim.conv2d(branch_3,96,[3,3])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.avg_pool2d(net,kernel_size=[3,3])
                    branch_4 = slim.conv2d(branch_4,32,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)
            #定义第一个模块组的第二个模块
            with tf.variable_scope("Module_1-2"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net, 64, [1, 1])
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net, 48, [1, 1])
                    branch_2 = slim.conv2d(branch_2, 64, [5, 5])
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.conv2d(net, 64, [1, 1])
                    branch_3 = slim.conv2d(branch_3, 96, [3, 3])
                    branch_3 = slim.conv2d(branch_3, 96, [3, 3])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.avg_pool2d(net, kernel_size=[3, 3])
                    branch_4 = slim.conv2d(branch_4, 64, [1, 1])
                net = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
            #定义第一个模块组的第三个模块，与第二个相同
            with tf.variable_scope("Module_1-3"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net, 64, [1, 1])
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net, 48, [1, 1])
                    branch_2 = slim.conv2d(branch_2, 64, [5, 5])
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.conv2d(net, 64, [1, 1])
                    branch_3 = slim.conv2d(branch_3, 96, [3, 3])
                    branch_3 = slim.conv2d(branch_3, 96, [3, 3])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.avg_pool2d(net, kernel_size=[3, 3])
                    branch_4 = slim.conv2d(branch_4, 64, [1, 1])
                net = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)

            #定义第二个模块组，包含5个模块
            #第一个模块
            with tf.variable_scope("Module_2-1"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,384,[3,3],stride=2,padding="VALID")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,64,[1,1])
                    branch_2 = slim.conv2d(branch_2,96,[3,3])
                    branch_2 = slim.conv2d(branch_2,96,[3,3],stride=2,padding="VALID")
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.max_pool2d(net,[3,3],stride=2,padding="VALID")
                net = tf.concat([branch_1,branch_2,branch_3],3)#输入35*35，输出17*17

            with tf.variable_scope("Module_2_2"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,192,[1,1],padding="SAME")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,128,[1,1])
                    branch_2 = slim.conv2d(branch_2,128,[1,7])
                    branch_2 = slim.conv2d(branch_2,192,[7,1])
                with tf.variable_scope("branch_3"):
                    branch_3= slim.conv2d(net,128,[1,1])
                    branch_3= slim.conv2d(branch_3,128,[7,1])
                    branch_3 = slim.conv2d(branch_3,128,[1,7])
                    branch_3 = slim.conv2d(branch_3,128,[7,1])
                    branch_3 = slim.conv2d(branch_3,192,[1,7])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.max_pool2d(net,[3,3])
                    branch_4= slim.conv2d(branch_4,192,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)
            with tf.variable_scope("Module_2_3"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,192,[1,1],padding="SAME")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,160,[1,1])
                    branch_2 = slim.conv2d(branch_2,160,[1,7])
                    branch_2 = slim.conv2d(branch_2,192,[7,1])
                with tf.variable_scope("branch_3"):
                    branch_3= slim.conv2d(net,128,[1,1])
                    branch_3= slim.conv2d(branch_3,160,[1,7])
                    branch_3 = slim.conv2d(branch_3,160,[7,1])
                    branch_3 = slim.conv2d(branch_3,160,[7,1])
                    branch_3 = slim.conv2d(branch_3,192,[1,7])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.max_pool2d(net,[3,3])
                    branch_4= slim.conv2d(branch_4,192,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)
            with tf.variable_scope("Module_2_4"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,192,[1,1],padding="SAME")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,160,[1,1])
                    branch_2 = slim.conv2d(branch_2,160,[1,7])
                    branch_2 = slim.conv2d(branch_2,192,[7,1])
                with tf.variable_scope("branch_3"):
                    branch_3= slim.conv2d(net,128,[1,1])
                    branch_3= slim.conv2d(branch_3,160,[1,7])
                    branch_3 = slim.conv2d(branch_3,160,[7,1])
                    branch_3 = slim.conv2d(branch_3,160,[7,1])
                    branch_3 = slim.conv2d(branch_3,192,[1,7])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.max_pool2d(net,[3,3])
                    branch_4= slim.conv2d(branch_4,192,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)
            with tf.variable_scope("Module_2_5"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,192,[1,1],padding="SAME")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,192,[1,1])
                    branch_2 = slim.conv2d(branch_2,192,[1,7])
                    branch_2 = slim.conv2d(branch_2,192,[7,1])
                with tf.variable_scope("branch_3"):
                    branch_3= slim.conv2d(net,192,[1,1])
                    branch_3= slim.conv2d(branch_3,192,[1,7])
                    branch_3 = slim.conv2d(branch_3,192,[7,1])
                    branch_3 = slim.conv2d(branch_3,192,[7,1])
                    branch_3 = slim.conv2d(branch_3,192,[1,7])
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.max_pool2d(net,[3,3])
                    branch_4= slim.conv2d(branch_4,192,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)

            #第三个模块组
            with tf.variable_scope("Module_3_1"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,192,[1,1])
                    branch_1 = slim.conv2d(branch_1,320,[3,3],stride=2,padding="VALID")
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,192,[1,1])
                    branch_2 = slim.conv2d(branch_2,192,[1,7])
                    branch_2 = slim.conv2d(branch_2,192,[7,1])
                    branch_2 = slim.conv2d(branch_2,192,[3,3],stride=2,padding="VALID")
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.max_pool2d(net,[3,3],stride=2,padding="VALID")
                net = tf.concat([branch_1,branch_2,branch_3],3)

            with tf.variable_scope("Module_3_2"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net,320,[1,1])
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net,384,[1,1])
                    branch_2_1 = slim.conv2d(branch_2,384,[1,3])
                    branch_2_2 = slim.conv2d(branch_2,384,[3,1])
                    branch_2 = tf.concat([branch_2_1,branch_2_2],3)
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.conv2d(net,384,[1,1])
                    branch_3_1 = slim.conv2d(branch_3,384,[1,3])
                    branch_3_2 = slim.conv2d(branch_3,384,[3,1])
                    branch_3 = tf.concat([branch_3_1,branch_3_2],3)
                with tf.variable_scope("branch_4"):
                    branch_4= slim.avg_pool2d(net,[3,3])
                    branch_4= slim.conv2d(branch_4,192,[1,1])
                net = tf.concat([branch_1,branch_2,branch_3,branch_4],3)

            with tf.variable_scope("Module_3_3"):
                with tf.variable_scope("branch_1"):
                    branch_1 = slim.conv2d(net, 320, [1, 1])
                with tf.variable_scope("branch_2"):
                    branch_2 = slim.conv2d(net, 384, [1, 1])
                    branch_2_1 = slim.conv2d(branch_2, 384, [1, 3])
                    branch_2_2 = slim.conv2d(branch_2, 384, [3, 1])
                    branch_2 = tf.concat([branch_2_1, branch_2_2], 3)
                with tf.variable_scope("branch_3"):
                    branch_3 = slim.conv2d(net, 384, [1, 1])
                    branch_3_1 = slim.conv2d(branch_3, 384, [1, 3])
                    branch_3_2 = slim.conv2d(branch_3, 384, [3, 1])
                    branch_3 = tf.concat([branch_3_1, branch_3_2], 3)
                with tf.variable_scope("branch_4"):
                    branch_4 = slim.avg_pool2d(net, [3, 3])
                    branch_4 = slim.conv2d(branch_4, 192, [1, 1])
                net = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
            return net

def inception_v3(inputs,dropout_keep_prob=0.5,scope = "InceptionV3"):
    net = inception_v3_base(inputs,scope)
    with tf.variable_scope("predict"):
        net = slim.avg_pool2d(net,[8,8],padding="VALID")
        net = slim.dropout(net,dropout_keep_prob)
        net = slim.conv2d(net,1000,[1,1],activation_fn=None)
        net = tf.squeeze(net,[1,2])
        predict = slim.softmax(net)
        return predict

def cal_run_time(sess,tensor,feed,info_string):
    iter_num = 10
    total_time = 0
    total_time_square = 0
    ave = 0
    for i in range(10 + iter_num):
        start = time.time()
        sess.run(tensor,feed_dict=feed)  # 执行tensor
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
    images = tf.Variable(tf.truncated_normal(shape=[16,299,299,3],stddev=0.1))#注意输入图片四维元素的含义和顺序
    dropout_keep_prob = tf.placeholder(tf.float32)
    sess = tf.InteractiveSession()
    predict = inception_v3(images,dropout_keep_prob)
    sess.run(tf.global_variables_initializer())
    cal_run_time(sess,predict,feed={dropout_keep_prob:0.8},info_string="前向运算")
