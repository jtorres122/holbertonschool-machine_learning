#!/usr/bin/env python3
'''Module contains the train function'''

import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop
tf.disable_v2_behavior()


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''Function builds, trains, and saves a neural network classifier'''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    model_save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            training_cost = sess.run(loss,
                                     feed_dict={x: X_train, y: Y_train})
            training_accuracy = sess.run(accuracy,
                                         feed_dict={x: X_train, y: Y_train})
            validation_cost = sess.run(loss,
                                       feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = sess.run(accuracy,
                                           feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(training_cost))
                print("\tTraining Accuracy: {}".format(training_accuracy))
                print("\tValidation Cost: {}".format(validation_cost))
                print("\tValidation Accuracy: {}".format(validation_accuracy))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        return model_save.save(sess, save_path)
