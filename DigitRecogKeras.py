import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

train_images = mnist_train_images.reshape(60000,784)
test_images = mnist_test_images.reshape(10000, 784)

#   Let's normalize our input data for faster convergence
#   The input image are all in 8-bit format - 8 bits gets 255, counting from 0.
#   So each pixel out of the 784 is is represented by an integer value
#   Between 0 and 255 inclusive.
#   We'll normalize these to a range of between 0 and 1 inclusive
#   First, need to convert to a floating-point type - so that we can get values
#   Between 0 and 1
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
#   And now we can divide by 255 to reduce the range to 0-1.
train_images /= 255
test_images /= 255

#   Now we'll convert the labels to one-hot format
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

def display_sample(num):
    print(train_labels[num])
    label = train_labels[num].argmax(axis=0)
    image = train_images[num].reshape([28,28])
    plt.title('Sample: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

display_sample(1234)

model = Sequential()    #   We can add layers to our NN one at a time - Sequentially
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test Loss: ', score[0])
print('Test accuracy: ', score[1])

for x in range(1000):
    test_image = test_images[x,:].reshape(1,784)
    predicted_cat = model.predict(test_image).argmax()
    label = test_labels[x].argmax()
    if(predicted_cat != label):
        plt.title('Prediction: %d Label: %d' % (predicted_cat, label))
        plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()