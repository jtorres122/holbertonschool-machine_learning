#!/usr/bin/env python3
'''Module contains train_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    '''
    Function trains a model using mini-batch gradient descent,
    also trains the model with learning rate decay
    '''
    if validation_data:
        callback = [K.callbacks.EarlyStopping(patience=patience)]
        lr_decay = K.callbacks.LearningRateScheduler(
            lambda epoch: alpha / (1 + decay_rate * epoch))
        callback.append(lr_decay)

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callback)
