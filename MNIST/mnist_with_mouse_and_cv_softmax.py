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

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict= \
                   {x: mnist.test.images, y_:mnist.test.labels}))

    while(1):
        cv2.imshow('test1', test1)    
        cv2.setMouseCallback('test1', drawByMouseForCV)
        test2 = cv2.resize(test1,(28, 28))
        #print(test2.shape)
        print(sess.run(tf.argmax(sess.run(y, feed_dict= \
                                          {x: test2.reshape([1,784])}) \
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


        
