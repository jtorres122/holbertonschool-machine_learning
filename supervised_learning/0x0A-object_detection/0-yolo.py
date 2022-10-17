#!/usr/bin/env python3
'''Module contains Yolo class'''
import tensorflow as tf


class Yolo():
    '''Yolo class'''

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''Class instatiator'''
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            line = [s.rstrip('\n') for s in f]
        self.class_names = line
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
