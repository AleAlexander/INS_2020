import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def createModel(optimizer):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def drawPlots(history):
    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test accuracy ')
    plt.plot(histoty.history['acc'], 'r', label='train')
    plt.plot(histoty.history['val_acc'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()

    plt.figure(1, figsize=(8, 5))
    plt.title('Training and test loss ')
    plt.plot(histoty.history['loss'], 'r', label='train')
    plt.plot(histoty.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()

def loadImage(filename):
    image = Image.open(filename)
    image = image.resize((28, 28))
    image = np.dot(np.asarray(image), np.array([1/3, 1/3, 1/3]))
    image /= 255
    image = 1 - image
    image = image.reshape((1, 28, 28))
    return image


if __name__ == '__main__':
    model = createModel('adam')
    histoty = model.fit(train_images, train_labels, epochs=5, batch_size=128,validation_data=(test_images, test_labels))
    drawPlots(histoty)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

#    image = loadImage('7.png')
 #   res = model.predict(image)
  #  print(np.argmax(res))
    print('test_acc:', test_acc)