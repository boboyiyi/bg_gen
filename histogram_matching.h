#ifndef HISTOGRAM_MATCHING_H
#define HISTOGRAM_MATCHING_H
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
class histogram_core {
private:
    uint8_t LUT[3][256];
    int source_hist_int[3][256];
    int target_hist_int[3][256];
    float source_histogram[3][256];
    float target_histogram[3][256];
public:
	histogram_core();
	~histogram_core();
    void histogram_matching(const cv::Mat &dst, const cv::Mat &src, const cv::Mat &mask, cv::Mat &cloned);
};
#endif