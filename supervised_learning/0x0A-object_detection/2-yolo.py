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
        boxes = []
        box_conf = []
        box_class = []
        boxes = [output[:, :, :, 0:4] for output in outputs]
        for oidx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    c_y = ((self.sigmoid(output[y, x, :, 1]) + y)
                           / output.shape[0] * image_size[0])
                    c_x = ((self.sigmoid(output[y, x, :, 0]) + x)
                           / output.shape[1] * image_size[1])
                    resize = self.anchors[oidx].astype(float)
                    resize[:, 0] *= (np.exp(output[y, x, :, 2])
                                     / 2 * image_size[1] /
                                     self.model.input.shape[1])
                    resize[:, 1] *= (np.exp(output[y, x, :, 3])
                                     / 2 * image_size[0] /
                                     self.model.input.shape[2])
                    output[y, x, :, 0] = c_x - resize[:, 0]
                    output[y, x, :, 1] = c_y - resize[:, 1]
                    output[y, x, :, 2] = c_x + resize[:, 0]
                    output[y, x, :, 3] = c_y + resize[:, 1]
        for output in outputs:
            box_conf.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class.append(self.sigmoid(output[..., 5:]))
        return (boxes, box_conf, box_class)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''Method filters boxes'''
        box_score = []
        bc = box_confidences
        bcp = box_class_probs
        for box_conf, box_probs in zip(bc, bcp):
            score = (box_conf * box_probs)
            box_score.append(score)
        box_classes = [s.argmax(axis=-1) for s in box_score]
        box_class_l = [b.reshape(-1) for b in box_classes]
        box_classes = np.concatenate(box_class_l)
        box_class_scores = [s.max(axis=-1) for s in box_score]
        b_scores_l = [b.reshape(-1) for b in box_class_scores]
        box_class_scores = np.concatenate(b_scores_l)
        mask = np.where(box_class_scores >= self.class_t)
        boxes_all = [b.reshape(-1, 4) for b in boxes]
        boxes_all = np.concatenate(boxes_all)
        scores = box_class_scores[mask]
        boxes = boxes_all[mask]
        classes = box_classes[mask]
        return (boxes, classes, scores)
