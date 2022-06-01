#!/usr/bin/env python3
'''Module contains train_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    '''
    Function trains a model using mini-batch gradient descent,
    also train the model using early stopping
    '''
    early_stopping_monitor = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=[early_stopping_monitor])
