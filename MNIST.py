from keras.utils import to_categorical
from keras import layers
from keras import models

""" (MNIST) It’s a set of 60,000 training
images, plus 10,000 test images, assembled by the National Institute of Standards and
Technology (the NIST in MNIST) in the 1980s. It's like the hello-world of programming in ML """

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
""" layer is a data-processing module that
you can think of as a filter for data. Some data goes in, and it comes out in a more useful form. Specifically, layers extract representations out of the data fed into them—hopefully, representations that are more meaningful for the problem at hand.  """
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
""" DENSE means the layers are densely connected (meanse that they are fully connected) """

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
""" 1: A loss function—How the network will be able to measure its performance on
the training data, and thus how it will be able to steer itself in the right direction.
2: An optimizer—The mechanism through which the network will update itself
based on the data it sees and its loss function.
3: Metrics to monitor during training and testing—Here, we’ll only care about accuracy(the fraction of the images that were correctly classified). """
""" Before training, we’ll preprocess the data by reshaping it into the shape the network
expects and scaling it so that all values are in the [0, 1] interval.  """
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
