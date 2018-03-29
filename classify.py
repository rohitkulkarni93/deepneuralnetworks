import tensorflow as tf
from keras.datasets import cifar10 as cf
import numpy as np
import sys
import cv2


def train_model_on_cifar10():
	# Hyperparameters
	beta = 0.009
	iters = 8000
	batch_size = 256
	learning_rate = 0.0000001
	image_size = 32 * 32 * 3
	classes = 10

	print('Learning rate: ', learning_rate, 'Regularizer beta: ', beta, 'Iteration: ', iters, 'Batch Size: ', batch_size)

	# Loading cifar10 dataset.
	(xTrain, yTrain), (xTest, yTest) = cf.load_data()
	xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
	xTest = np.reshape(xTest, (xTest.shape[0], -1))
	yTrain = np.squeeze(yTrain)
	yTest = np.squeeze(yTest)

	# Create the model
	x = tf.placeholder(tf.float32, [None, image_size], 'x')
	y_ = tf.placeholder(tf.int64, [None], 'expected_output')

	# Variables
	W = tf.Variable(tf.zeros([image_size, classes], name='weights'))
	b = tf.Variable(tf.zeros([classes]), name='bias')

	# Output
	y1 = tf.matmul(x, W)
	y = tf.add(y1, b, name='y')

	correct_prediction = tf.equal(tf.argmax(y, 1), y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()
	
	# Define loss and optimizer
	loss_l = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, classes), y))
	regularizer = tf.nn.l2_loss(W)
	loss_l = tf.reduce_mean(loss_l + beta * regularizer)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_l)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	print('Training model.')
	
	# Train
	for step in range(iters):
		s = np.arange(xTrain.shape[0])
		np.random.shuffle(s)
		xTr = xTrain[s]
		yTr = yTrain[s]
		batch_xs = xTr[:batch_size]
		batch_ys = yTr[:batch_size]
		train_loss, _, train_acc = sess.run([loss_l, train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
		test_loss, test_acc = sess.run([loss_l, accuracy], feed_dict={x: xTest, y_: yTest})
		if step % 10 == 0:
			print('Iteration: {0} Train Loss: {1:.5f} Train Acc % {2:.2f} Test Loss {3:.5f} Test Acc: {4:.2f} %'
				.format(step, train_loss, train_acc * 100, test_loss, test_acc * 100))

	save_path = saver.save(sess, "./model/model.ckpt")

	# Test model and print accuracy.
	print('Testing Accuracy: ', sess.run(accuracy, feed_dict={x: xTest, y_: yTest}) * 100)

def predict(img_data):

	class_labels = list(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck'])

	sess = tf.Session()
	saver =  tf.train.import_meta_graph('./model/model.ckpt.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./model/'))
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name('x:0')
	y = graph.get_tensor_by_name('y:0')
	prediction = tf.argmax(y, 1)
	answer = prediction.eval(session=sess, feed_dict={x : img_data})
	print(class_labels[answer[0]])


def load_input_image(img):
	img = cv2.imread(img, 1)
	res = cv2.resize(img,(32, 32))
	res=np.array(res).reshape(1, 32,32,3)
	res = np.reshape(res, (res.shape[0], -1))
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
