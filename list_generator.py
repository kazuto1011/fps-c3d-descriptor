# -*- coding: utf-8 -*-

if __name__ == '__main__':
    f1 = open('input_list_video.txt', 'w')
    f2 = open('output_list_prefix.txt', 'w')
    act_list = ["read-a-book", "eat-a-meal", "gaze-at-a-robot", "gaze-at-a-tree", "look-around"]

    for a in act_list:
        for i in range(0, 50):
            for j in range(0, 32):
                s = "input/%s/%s_%d.avi %d 0\n" % (a, a, i, 8 * j)
                f1.write(s)

    for a in act_list:
        for i in range(0, 50):
            for j in range(0, 32):
                s = "output/%s/%s_%d_%d\n"%(a, a, i, 8*j)
                f2.write(s)

    f1.close()
    f2.close()
