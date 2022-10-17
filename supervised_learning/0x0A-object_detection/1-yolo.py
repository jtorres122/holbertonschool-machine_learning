#!/usr/bin/env python3
'''Module contains Yolo class'''
import tensorflow as tf
import numpy as np


class Yolo():
    '''Yolo class'''

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''Class instantiator'''
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            line = [s.rstrip('\n') for s in f]
        self.class_names = line
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        '''Method calculates sigmoid function'''
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        '''Method processes outputs'''
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(pred[grid_h, grid_w, :, 0]) + grid_w)
                          / pred.shape[1])
                    by = ((self.sigmoid(pred[grid_h, grid_w, :, 1]) + grid_h)
                          / pred.shape[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= np.exp(pred[grid_h, grid_w, :, 2])\
                        / self.model.input.shape[1].value
                    anchor_tensor[:, 1] *= np.exp(pred[grid_h, grid_w, :, 3])\
                        / self.model.input.shape[2].value
                    pred[grid_h, grid_w, :, 0] = (bx - (anchor_tensor[:, 0] /
                                                  2)) * image_size[1]
                    pred[grid_h, grid_w, :, 1] = (by - (anchor_tensor[:, 1] /
                                                  2)) * image_size[0]
                    pred[grid_h, grid_w, :, 2] = (bx + (anchor_tensor[:, 0] /
                                                  2)) * image_size[1]
                    pred[grid_h, grid_w, :, 3] = (by + (anchor_tensor[:, 1] /
                                                  2)) * image_size[0]
        box_confidences = [self.sigmoid(pred[:, :, :, 4:5]) for pred in
                           outputs]
        box_class_probs = [self.sigmoid(pred[:, :, :, 5:]) for pred in outputs]
        return (boxes, box_confidences, box_class_probs)
