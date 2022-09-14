#!/usr/bin/env python3
'''Module contains optimize_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    '''Function trains a model using mini-batch gradient descent'''

    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            '''
            gets the learning rate of each epoch
            :epoch: is the current epoch
            '''
            return alpha / (1 + decay_rate * epoch)

        callbacks = []
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
        callbacks.append(K.callbacks.LearningRateScheduler(
                         scheduler, verbose=1))
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, callbacks=[callbacks],
                           validation_data=validation_data,
                           verbose=verbose, shuffle=shuffle)
    else:
        return network.fit(x=data, y=labels, batch_size=batch_size,
                           epochs=epochs, validation_data=validation_data,
                           verbose=verbose, shuffle=shuffle)
