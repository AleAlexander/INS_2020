from tensorflow.keras.callbacks import Callback
from datetime import datetime

class MyCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.date = datetime.now()
        self.name = 'model_number'
        self.prefix = f'{self.date.day}_{self.date.month}_{self.date.year}_{self.name}_'
        self.loss = {}
        self.key = 'val_loss'
        self.index = 0


    def on_train_begin(self, logs=None):
        loss, _ = self.model.evaluate(self.validation_data[0], self.validation_data[1])
        for i in range(1,4):
            self.loss[self.prefix + str(i)] = loss
        for key in self.loss.keys():
            self.model.save(key)

    def on_epoch_end(self, epoch, logs=None):
        for i in range(1, 4):
            if logs.get(self.key) < self.loss[self.prefix + str(i)] and i > self.index:
                self.loss[self.prefix + str(i)] = logs.get(self.key)
                self.model.save(self.prefix + str(i))
                self.index += 1
                break
            elif i <= self.index:
                continue
        if (self.index == 3):
            self.index = 0