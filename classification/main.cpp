#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;
using namespace std;

#define NUM_CLIP 32
#define NUM_VIDEO 50
#define REPETITON 100

void count_up(cv::Mat src,cv::Mat& dst){
    cv::Mat t_dst = cv::Mat::zeros(cv::Size(src.cols, 1), src.type());
    float* dst_row = t_dst.ptr<float>(0);
    for (int i = 1; i < src.rows; i++)
    {
        float* src_prerow = src.ptr<float>(i-1);
        float* src_row = src.ptr<float>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if(src_row[j] > src_prerow[j])
                dst_row[j]++;
        }
    }
    dst = t_dst;
}

void count_down(cv::Mat src,cv::Mat& dst){
    cv::Mat t_dst = cv::Mat::zeros(cv::Size(src.cols, 1), src.type());
    float* dst_row = t_dst.ptr<float>(0);
    for (int i = 1; i < src.rows; i++)
    {
        float* src_prerow = src.ptr<float>(i-1);
        float* src_row = src.ptr<float>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if(src_row[j] < src_prerow[j])
                dst_row[j]++;
        }
    }
    dst = t_dst;
}

void count_updown(cv::Mat src,cv::Mat& dst){
    cv::Mat t_dst = cv::Mat::zeros(cv::Size(src.cols*2, 1), src.type());
    float* dst_u = t_dst.ptr<float>(0);
    float* dst_d = dst_u + src.cols;
    for (int i = 1; i < src.rows; i++)
    {
        float* src_prerow = src.ptr<float>(i-1);
        float* src_row = src.ptr<float>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if(src_row[j] > src_prerow[j])
                dst_u[j]++;
            else if(src_row[j] < src_prerow[j])
                dst_d[j]++;
        }
    }
    dst = t_dst;
}

int main()
{
    vector<string> act_list;
    act_list.push_back("read-a-book");
    act_list.push_back("eat-a-meal");
    act_list.push_back("gaze-at-a-robot");
    act_list.push_back("gaze-at-a-tree");
    act_list.push_back("look-around");
    string ext = ".fc6-1";
    //string ext = ".fc7-1";
    //string ext = ".prob";

    srand((unsigned)time(NULL));

    vector<int> index_list;
    for (int i = 0; i < NUM_VIDEO; i++)
        index_list.push_back(i);

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    float correct_ratio = 0.f;
    for (int r = 0; r < REPETITON; r++)
    {
        cv::Mat train, test;
        vector<int> train_label, test_label;

        int correct_label = 0;
        for (string act : act_list)
        {
            std::random_shuffle(index_list.begin(), index_list.end());
            for (int i = 0; i < NUM_VIDEO/2; i++)
            {
                cv::Mat video;
                for (int j = 0; j < NUM_CLIP; j++)
                {
                    fs::path feature_file = "/home/kazuto/fourth_person_sensing/C3D/C3D_video_descriptor/output";
                    feature_file /= act;
                    feature_file /= act;
                    feature_file += "_";
                    feature_file += to_string(index_list[i]);
                    feature_file += "_";
                    feature_file += to_string(8*j);
                    feature_file += ext;

                    FILE *f;
                    f = fopen(feature_file.c_str(), "rb");
                    if (f == NULL)
                        return false;

                    int n, c, l, w, h;
                    fread(&n, sizeof(int), 1, f);
                    fread(&c, sizeof(int), 1, f);
                    fread(&l, sizeof(int), 1, f);
                    fread(&h, sizeof(int), 1, f);
                    fread(&w, sizeof(int), 1, f);

                    int dimension = n*c*l*h*w;
                    float * buff = new float[dimension];
                    fread(buff, sizeof(float), dimension, f);

                    cv::Mat feat_vec = cv::Mat(1, dimension, CV_32FC1);
                    float* vec_row = feat_vec.ptr<float>(0);
                    for (int i = 0; i < dimension; i++)
                        vec_row[i] = buff[i];
                    video.push_back(feat_vec);
                    fclose(f);
                }
                cv::reduce(video, video, 0, CV_REDUCE_AVG);
                //cv::reduce(video, video, 0, CV_REDUCE_MAX);
                //count_up(video, video);
                //count_down(video, video);
                //count_updown(video, video);
                cv::normalize(video,video,cv::NORM_L2SQR);
                train.push_back(video);
                train_label.push_back(correct_label);
            }

            for (int i = NUM_VIDEO/2; i < NUM_VIDEO; i++)
            {
                cv::Mat video;
                for (int j = 0; j < NUM_CLIP; j++)
                {
                    fs::path feature_file = "/home/kazuto/fourth_person_sensing/C3D/C3D_video_descriptor/output";
                    feature_file /= act;
                    feature_file /= act;
                    feature_file += "_";
                    feature_file += to_string(index_list[i]);
                    feature_file += "_";
                    feature_file += to_string(8*j);
                    feature_file += ext;

                    FILE *f;
                    f = fopen(feature_file.c_str(), "rb");
                    if (f == NULL)
                        return false;

                    int n, c, l, w, h;
                    fread(&n, sizeof(int), 1, f);
                    fread(&c, sizeof(int), 1, f);
                    fread(&l, sizeof(int), 1, f);
                    fread(&h, sizeof(int), 1, f);
                    fread(&w, sizeof(int), 1, f);

                    int dimension = n*c*l*h*w;
                    float * buff = new float[dimension];
                    fread(buff, sizeof(float), dimension, f);

                    cv::Mat feat_vec = cv::Mat(1, dimension, CV_32FC1);
                    float* vec_row = feat_vec.ptr<float>(0);
                    for (int i = 0; i < dimension; i++)
                        vec_row[i] = buff[i];
                    video.push_back(feat_vec);
                    fclose(f);
                }
                cv::reduce(video, video, 0, CV_REDUCE_AVG);
                //cv::reduce(video, video, 0, CV_REDUCE_MAX);
                //count_up(video, video);
                //count_down(video, video);
                //count_updown(video, video);
                cv::normalize(video,video,cv::NORM_L2SQR);
                test.push_back(video);
                test_label.push_back(correct_label);
            }
            correct_label++;
        }

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#if 1 //PCA
        int pre_dim = train.cols;
        cv::PCA pca(train, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.95);
        pca.project(train, train);
        pca.project(test, test);
        cout << "PCA 95%: " << pre_dim << " >> " << train.cols << endl;
#endif
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        cv::SVM svm;
        cv::SVMParams params;
        params.svm_type = cv::SVM::C_SVC;
        params.kernel_type = cv::SVM::LINEAR;

        cv::Mat label(train.rows, 1, CV_32SC1);

        for (int i = 0; i < train.rows; i++)
            label.at<int>(i, 0) = train_label[i];

        svm.train(train, label, cv::Mat(), cv::Mat(), params);

        int accuracy = 0;
        for (int i = 0; i < test.rows; i++)
        {
            float res = svm.predict(test.row(i));
            if ((int)res == test_label[i])
                accuracy++;
        }

        //printf("Correct Classification Rate is %lf\n", (double)accuracy / (double)train.rows);
        correct_ratio += (double)accuracy / (double)test.rows;
#if 1
        cv::imshow("features", test*2);
        cv::waitKey(10);
#endif
    }

    printf("Correct Classification Rate is %lf\n", correct_ratio/REPETITON);

    return 0;
}
