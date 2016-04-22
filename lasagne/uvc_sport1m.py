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

# Build model
net = c3d.build_model()

# Set the weights (takes some time)
c3d.set_weights(net['prob'],'c3d_model.pkl')

# Sample clip
snip=np.load('example_snip.npy')
caffe_snip=c3d.get_snips(snip,image_mean=np.load('snipplet_mean.npy'),start=0, with_mirrored=False)

prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
pred_fn = theano.function([net['input'].input_var], prediction, allow_input_downcast = True);

class Sport1MClassifier():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.clip = np.ndarray((1,3,16,112,112),dtype=float)
        cv2.namedWindow("camera")

    def run(self):
        while True:
            self.retrieve_clips()
            self.classfy()

    def retrieve_clips(self):
        clip = []
        for _ in range(16):
            ret, frame = self.cap.read()
            cv2.imshow("camera", frame)
            cv2.waitKey(10)
            frame = cv2.resize(frame, (112,112))
            clip.append(frame)
        self.clip = np.asarray([clip]).transpose(0,4,1,2,3)

    def classfy(self):
        probabilities=pred_fn(self.clip).mean(axis=0)
        # Load labels
        with open('labels.txt','r') as f:
            class2label=dict(enumerate([name.rstrip('\n') for name in f]))
        # Show the post probable ones
        print('Top 10 class probabilities:')
        for class_id in (-probabilities).argsort()[0:10]:
            print('%20s: %.2f%%' % (class2label[class_id],100*probabilities[class_id]))

classifier = Sport1MClassifier()
classifier.run()

