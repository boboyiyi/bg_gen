#ifndef ALIGNER_H
#define ALIGNER_H
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifdef __linux__
#define delimiter "/"
#define creat_dir "mkdir -p "
#elif _WIN32
#define delimiter "\\"
#define creat_dir "mkdir "
#else
#endif

#define NUM_PTS 68
#define NUM_ALIGN_PTS 51

class aligner {
public:
	aligner();
	~aligner();
    void pts_read(std::string &, Eigen::MatrixXf &point);
    void pts_write(std::string &, std::vector<cv::Point2f> &points);
    Eigen::MatrixXf umeyama(std::string &, std::string &);
    Eigen::MatrixXf ls(std::string &, std::string &);
};
#endif