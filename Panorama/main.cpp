#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <opencv2/opencv.hpp>
#include "mystitcher.h"

using namespace std;
using namespace cv;

int main() {
	vector<Mat> images;
	for (int i = 0; i < 13; ++i) {
		images.push_back(imread(("images/belf" + to_string(i + 1) + ".jpg").c_str()));
	}
	MyStitcher stitcher;
	Mat panorama = stitcher.stitch(images);
	imwrite("panorama.jpg", panorama);
	return 0;
}