#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2016-04-22

recipes_dir = "/home/common/Lasagne/Recipes"

import os.path as osp
import sys
sys.path.insert(0,osp.join(recipes_dir, "modelzoo"))
import lasagne
import c3d
import theano
import numpy as np
import cv2

class Sport1MClassifier():
    def __init__(self):
        self.result = ["",""]
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow("camera")

        # Build model
        self.net = c3d.build_model()
        c3d.set_weights(self.net['prob'],'c3d_model.pkl')

        self.prediction = lasagne.layers.get_output(self.net['prob'], deterministic=True)
        self.pred_fn = theano.function([self.net['input'].input_var], self.prediction, allow_input_downcast=True);

        # Load labels
        with open('labels.txt','r') as f:
            self.cls2label = dict(enumerate([name.rstrip('\n') for name in f]))

    def run(self):
        while True:
            clip = self.retrieve_clips()
            self.classfy(clip)

    def retrieve_clips(self):
        frames = []
        for _ in range(16):
            ret, frame = self.cap.read()
            frames.append(cv2.resize(frame, (112,112)))
            cv2.putText(frame, self.result[0], (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(frame, self.result[1], (10,70), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow("camera", frame)
            cv2.waitKey(10)
        return np.asarray([frames]).transpose(0,4,1,2,3)

    def classfy(self, clip):
        self.cls_score = self.pred_fn(clip).mean(axis=0)
        
        top_id = (-self.cls_score).argsort()[0:10]
        self.result = self.cls2label[top_id[0]], "%.2f%%"%(100*self.cls_score[top_id[0]])
        
        print('Top 10 class probabilities:')
        for cls_id in top_id:
            print('%20s: %.2f%%' % (self.cls2label[cls_id],100*self.cls_score[cls_id]))

classifier = Sport1MClassifier()
classifier.run()

