#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	Mat src = imread("WoW1.jpg");
	imshow("original", src);
	waitKey();
	return 0;
}