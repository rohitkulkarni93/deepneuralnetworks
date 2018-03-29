import tensorflow as tf
from keras.datasets import cifar10 as cf
import numpy as np
import sys
import cv2
import os
import math
import matplotlib as mp
import matplotlib.pyplot as plt


def train_model_on_cifar10():
    
    learning_rate = 5e-4
    epochs = 2500
    batch_size = 256
    
    # Loading cifar10 dataset.
    (xTrain, yTrain), (xTest, yTest) = cf.load_data()
    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)
    
    # Create model
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
    print(x.name)
    y = tf.placeholder(tf.int64, [None], 'expected_output')
    
    # 2 Convolution layers with activation function RELU.
    conv_layer1 = tf.layers.conv2d(inputs=x, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    conv_layer2 = tf.layers.conv2d(inputs=conv_layer1, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    
    # 1 Max pooling layer
    max_pooling_layer1 = tf.layers.max_pooling2d(inputs=conv_layer2, pool_size=[2, 2], strides=2)
    
    # 2 Convolution layers with activation function RELU.
    conv_layer3 = tf.layers.conv2d(inputs=max_pooling_layer1, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    conv_layer4 = tf.layers.conv2d(inputs=conv_layer3, filters = 64, padding = 'same', kernel_size=5, strides = 1, activation=tf.nn.relu)
    
    # 1 Max pooling layer
    max_pooling_layer2 = tf.layers.max_pooling2d(inputs=conv_layer4, pool_size=[2, 2], strides=2)
    
    # 1 Convolution layer with activation function as RELU
    conv_layer5 = tf.layers.conv2d(inputs=max_pooling_layer2, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    
    # 1 Max pooling
    max_pooling_layer3 = tf.layers.max_pooling2d(inputs=conv_layer5, pool_size=[2, 2], strides=2)
    
    # Flatten to provide input to fully connected layer
    reshape_vector = tf.reshape(max_pooling_layer3, [-1, 1024])
    fully_connected_layer1 = tf.layers.dense(inputs=reshape_vector, units=1024, activation=tf.nn.relu)
    
    # fully connected layer
    y_out = tf.layers.dense(inputs=fully_connected_layer1, units=10, activation=None, name='y_out')
    print(y_out.name)
    # Define Loss
    total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
    mean_loss = tf.reduce_mean(total_loss)
    
    # Define Optimizer
    adam_optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = adam_optimizer.minimize(mean_loss)
    
    # Define correct Prediction and accuracy
    correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    trainIndex = np.arange(xTrain.shape[0])
    
    with tf.Session() as sess:
        print("Loop\t\tTrain Loss\t\tTrain Acc %\t\tTest loss\t\tTest Acc %")
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            # Shuffle input like last time.
            s = np.arange(xTrain.shape[0])
            np.random.shuffle(s)
            xTr = xTrain[s]
            yTr = yTrain[s]
            batch_xs = xTr[:batch_size]
            batch_ys = yTr[:batch_size]
            train_loss, train_acc, _ = sess.run([mean_loss, accuracy, train_step], feed_dict={x: batch_xs, y: batch_ys })
            test_loss = []
            test_acc = []
            # Split test input in 4 batchs to avoid ResourceExhaustedError exception in GPU.
            size = xTest.shape[0] / 4
            for i in range(4):
                start = int(size * i)
                end = int(size * (i + 1) - 1)
                tl, ta = sess.run([mean_loss, accuracy], feed_dict={x: xTest[start:end], y: yTest[start:end]})
                test_loss.append(tl)
                test_acc.append(ta)
            print('{0}\t\t{1:0.6f}\t\t{2:0.6f}\t\t{3:0.6f}\t\t{4:0.6f}'.format(int(epoch), train_loss, train_acc * 100, sum(test_loss) / 4, (sum(test_acc) / 4) * 100))
        #save session
        save_path = saver.save(sess, './model/cifar')
        print("Model saved in file: ", save_path)
    sess.close()


def plot_filter(units):
    filters = 32
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig('CONV_rslt' + '.png')

def predict(img_data):
    
    class_labels = list(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                         'horse', 'ship', 'truck'])
        
    
    sess = tf.Session()
    saver =  tf.train.import_meta_graph('./model/cifar.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    #y = graph.get_tensor_by_name('output_y:0')
    conv = graph.get_tensor_by_name('conv2d/Relu:0')
    y_out = graph.get_tensor_by_name('y_out/BiasAdd:0')
    #sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    #prediction = tf.argmax(y_out, 1)

    test_loss = sess.run(y_out, feed_dict={x : img_data})
    print(class_labels[np.argmax(test_loss)])

    layer_to_vis = sess.run(conv, feed_dict={x: img_data})
    plot_filter(layer_to_vis)
    
    #visualize graph
    #plt.imshow(img_data, interpolation="nearest", cmap="gray")

def load_input_image(img):
    img = cv2.imread(img, 1)
    res = cv2.resize(img,(32, 32))
    res=np.array(res).reshape(1, 32, 32, 3)
    #res = np.reshape(res, (res.shape[0], -1))
    return res

def predict_image_class(image):
    # Load the image data.
    img_data = load_input_image(image)
    # Load the model and predict class.
    predict(img_data)
# Predict class.

if len(sys.argv) == 2:
    if sys.argv[1] == 'train':
        train_model_on_cifar10()
elif len(sys.argv) == 3:
    if sys.argv[1] == 'test':
        image = sys.argv[2]
        predict_image_class(image)
