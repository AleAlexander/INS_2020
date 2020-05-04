import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import string


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def loadData(dimension):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")
    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (train_x, train_y), (test_x, test_y)

def drawPlots(history):
    x = range(1, num_epochs + 1)
    plt.plot(x, history.history['loss'], label='Training loss')
    plt.plot(x, history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(x, history.history['acc'], label='Training acc')
    plt.plot(x, history.history['val_acc'], label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def createModel():
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def testText(filename):
    data = loadText(filename)
    data = vectorize([data], 10000)
    res = model.predict(data)
    print(res)

def loadText(filename):
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    stripped_low = []
    for w in stripped:
        stripped_low.append(w.lower())
    print(stripped_low)
    indexes = imdb.get_word_index()
    encoded = []
    for w in stripped_low:
        if w in indexes and indexes[w] < 10000:
            encoded.append(indexes[w])
    data = np.array(encoded)
    return data
    

if __name__ == '__main__':
    dimension = 10000
    num_epochs = 2

    (train_x, train_y), (test_x, test_y) = loadData(dimension)
    model = createModel()
    history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=500, validation_data=(test_x, test_y))
    testText("negative.txt")
    # drawPlots(history) 