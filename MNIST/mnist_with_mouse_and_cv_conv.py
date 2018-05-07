import read_data
import cv2
import tensorflow as tf
import numpy as np


old_x, old_y, cur_x, cur_y = 0, 0, 0, 0
mouse_flag = False


def getImage(mnist, ind):
    return mnist.test.images[ind].reshape([28,28])


def drawByMouseForCV(event, x, y, flags, param):
    global old_x, old_y, mouse_flag, test1
    if event == cv2.EVENT_LBUTTONDOWN:
        if mouse_flag == False:
            old_x, old_y = x, y
            cv2.circle(test1, (x, y), 3, (255, 255, 255), -1)
            mouse_flag = True
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_flag == True:
            cv2.circle(test1, (x, y), 3, (255, 255, 255), -1)
            #print(str(old_x) + " " + str(old_y))
            cv2.line(test1, (old_x, old_y), (x, y), \
                     (255, 255, 255), thickness=6)
            old_x, old_y = x, y
    if event == cv2.EVENT_LBUTTONUP:
        if mouse_flag == True:
            mouse_flag = False
    if event == cv2.EVENT_RBUTTONDOWN:
        test1 = np.zeros((280, 280, 1), np.uint8)
            

mnist= read_data.read_data_sets("C:\\workspace\\MNIST", one_hot = True)

#print(mnist.count)
#print(mnist.train.images.shape)
ind=300
#test1 = mnist.test.images[ind].reshape([28,28])
test1 = np.zeros((280, 280, 1), np.uint8)
#print(mnist.train.labels[ind])


#Test the datasets....ABOVE!!


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding='SAME')


#Input layer
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


#2 convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Full connection layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#accuracy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={ \
                x: batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], \
                                        keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={ \
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    while(1):
        cv2.imshow('test1', test1)    
        cv2.setMouseCallback('test1', drawByMouseForCV)
        test2 = cv2.resize(test1,(28, 28))
        #print(test2.shape)
        print(sess.run(tf.argmax(sess.run(y_conv, feed_dict= \
                                          {x: test2.reshape([1,784]), \
                                           keep_prob: 1.0}) \
                    , 1)))
        cv2.imshow('test2', test2)
        #cv2.resizeWindow('test1', 280,280)
            
        if cv2.waitKey(20) >0:
            break;

    cv2.destroyAllWindows()

    #print(sess.run(tf.argmax(sess.run(y, feed_dict= \
    #                         {x: mnist.test.images[ind].reshape([1,784])}) \
    #                , 1)))
    #print(sess.run(tf.argmax(mnist.test.labels[ind], 0)))


        
