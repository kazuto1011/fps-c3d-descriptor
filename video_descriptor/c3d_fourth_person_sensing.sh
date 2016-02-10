mkdir -p output/eat-a-meal
mkdir -p output/gaze-at-a-robot
mkdir -p output/gaze-at-a-tree
mkdir -p output/look-around
mkdir -p output/read-a-book
GLOG_logtosterr=1 ~/C3D/build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt ./conv3d_deepnetA_sport1m_iter_1900000 -1 50 160 ./prototxt/output_list_prefix.txt fc7-1 fc6-1 prob
