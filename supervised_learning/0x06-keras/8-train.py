#!/usr/bin/env python3
'''Module contains optimize_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    '''Function trains a model using mini-batch gradient descent'''

    def scheduler(epoch):
        '''
        gets the learning rate of each epoch
        :epoch: is the current epoch
        '''
        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        callbacks = []
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(
                             scheduler, verbose=1))
        if save_best:
            callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath))
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, callbacks=[callbacks],
                           validation_data=validation_data,
                           verbose=verbose, shuffle=shuffle)
    else:
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, validation_data=validation_data,
                           verbose=verbose, shuffle=shuffle)
