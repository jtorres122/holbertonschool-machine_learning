#!/usr/bin/env python3
'''Module contains preprocess_data function and buils a model'''
import tensorflow.keras as K


def preprocess_data(X, Y):
    '''Function pre-processes the data for the model'''

    X_p = K.applications.resnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(x_train, y_train)
    X_test_p, Y_test_p = preprocess_data(x_test, y_test)

    resnet_base = K.applications.ResNet50(
        include_top=False,
        input_shape=(224, 224, 3)
    )

    input_layer = K.Input(shape=(32, 32, 3))
    resizing_layer = K.layers.Lambda(
        lambda img:
        K.preprocessing.image.smart_resize(
            img, (224, 224)))(input_layer)

    resnet_layer = resnet_base(resizing_layer, training=False)
    flatten_layer = K.layers.Flatten()(resnet_layer)
    d1_layer = K.layers.Dense(500, activation='relu')(flatten_layer)
    dropout_layer = K.layers.Dropout(0.3)(d1_layer)
    output_layer = K.layers.Dense(10, activation='softmax')(dropout_layer)
    model = K.Model(inputs=input_layer, outputs=output_layer)

    model.summary()
    resnet_base.trainable = False

    model.compile(K.optimizers.Adam(learning_rate=.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
            X_train_p,
            Y_train_p,
            validation_data=(X_test_p, Y_test_p),
            batch_size=300,
            epochs=4,
            verbose=1)

    results = model.evaluate(X_test_p, Y_test_p)
    model.save('cifar10.h5')
