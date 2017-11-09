import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def display_sample(num):
    #Print one-hot array:
    print(mnist.train.labels[num])
    #Print the actual label
    label = mnist.train.labels[num].argmax(axis=0)
    #Set up the image and show it
    image = mnist.train.images[num].reshape([28,28])
    plt.title('Label: %d Sample: %d' % (label, num))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
display_sample(1234)

images = mnist.train.images[0].reshape([1,784])  #   Reshaping into input form
for i in range(1,500):
    images = np.concatenate((images, mnist.train.images[i].reshape([1,784])))
plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()

input_images = tf.placeholder(tf.float32, shape=[None,784])
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

#   Set up the nodes/network architecture
hidden_nodes = 1024

input_weights = tf.Variable(tf.truncated_normal([784,hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

#   Begin forward propagation using ReLU
input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

#   Learning the weights begins with the loss/cost function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))

#   Optimization method for our cost function
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#   Now let's give some metrics to measure the accuracy of our model
correct_predictions = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#   Now let's run 2000 iterations in batches of 100 examples
#   Every 100th epoch, print information regarding accuracy
tf.global_variables_initializer().run()

for x in range(2000):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0], target_labels:batch[1]})
    if((x+1) % 100 == 0):
        print("Training Epoch " + str((x+1)))
        print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels: mnist.test.labels})))



#   Now, we'll give ourselves a chance to examine any misclassified samples
for x in range(100):
    # Load a single test image and its label
    x_train = mnist.test.images[x,:].reshape(1,784)
    y_train = mnist.test.labels[x,:]
    # Convert the one-hot label into an integer
    label = y_train.argmax()
    # Get the classification from our NN's digit_weights layer and convert to int
    prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()
    # If the prediction does not match the target label, disply sample info
    if(prediction != label):
        plt.title('Prediction: %d Label: %d' % (prediction,label))
        plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()