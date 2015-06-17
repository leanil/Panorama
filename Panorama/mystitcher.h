#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class MyStitcher
{
public:
	MyStitcher(int focal = 500) :focal{ focal } {}
	cv::Mat stitch(const std::vector<cv::Mat>& images);
	cv::Mat cylindrical_warp(const cv::Mat& image, int focal, bool crop = true);
	long long SSD_error(const cv::Mat& a, const cv::Mat& b);
private:
	int focal;
};