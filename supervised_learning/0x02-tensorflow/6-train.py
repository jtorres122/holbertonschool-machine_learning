#!/usr/bin/env python3
'''Module contains train function'''
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''Trains a neural network classifier'''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    accuracy = calculate_accuracy(y, y_pred)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            train_loss, _, train_acc =\
                sess.run([loss, train_op, accuracy],
                         feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == iterations:
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: X_valid,
                                                        y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_loss))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(val_loss))
                print("\tValidation Accuracy: {}".format(val_acc))
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
