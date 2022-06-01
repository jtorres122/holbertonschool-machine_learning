#!/usr/bin/env python3
'''Module contains train_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    '''
    Function trains a model using mini-batch gradient descent,
    also save the best iteration of the model
    '''
    early_stopping_monitor = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
    callback = [early_stopping_monitor]

    if validation_data and learning_rate_decay:
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=lambda epoch: alpha / (1 + decay_rate * epoch), verbose=1)
        callback.append(lr_decay)

    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                    save_best_only=save_best))

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callback)
