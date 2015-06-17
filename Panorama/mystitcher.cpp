#include "mystitcher.h"

#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace cv;

Mat MyStitcher::stitch(const vector<Mat>& images) {
	Mat warped = cylindrical_warp(images[0], focal);
	Mat panorama(warped.rows, (int)images.size()*warped.cols, warped.type());
	warped.copyTo(panorama.colRange(0, warped.cols));
	int last_col = warped.cols, overlap = warped.cols / 6;
	for (unsigned i = 1; i < images.size(); ++i) {
		warped = cylindrical_warp(images[i], focal);
		int best_shift = 0;
		long long min_error = -1;
		for (int shift = overlap; shift <= warped.cols; ++shift) {
			long long tmp = SSD_error(panorama.colRange(last_col - shift, last_col), warped.colRange(0, shift));
			if (min_error < 0 || tmp < min_error) {
				min_error = tmp;
				best_shift = shift;
			}
		}
		last_col += warped.cols - best_shift;
		warped.copyTo(panorama.colRange(last_col - warped.cols, last_col), warped);
	}
	return panorama.colRange(0, last_col);
}

Mat MyStitcher::cylindrical_warp(const Mat& image, int focal, bool crop) {
	Mat result(image.size(), image.type());
	int xc = image.cols / 2, yc = image.rows / 2;
	int top, left = 0, bottom, right = image.cols;
	for (int x = 0; x < result.cols; ++x) {
		for (int y = 0; y < result.rows; ++y) {
			int sx = (int)round(focal*tan((double)(x - xc) / focal)) + xc;
			int sy = (int)round(double(y - yc) / cos((double)(x - xc) / focal)) + yc;
			if (sy >= 0 && sy < image.size().height && sx >= 0 && sx < image.size().width) {
				result.at<Vec3b>(y, x) = image.at<Vec3b>(sy, sx);
			}
			else {
				result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
				if (y == yc) {
					if (x < xc) { left = max(left, x + 1); }
					else { right = min(right, x); }
				}
			}
		}
	}
	for (top = 0; result.at<Vec3b>(top, left) == Vec3b(0, 0, 0); ++top);
	for (bottom = result.rows; result.at<Vec3b>(bottom - 1, left) == Vec3b(0, 0, 0); --bottom);
	return crop ? result(Range(top, bottom), Range(left, right)) : result;
}

long long MyStitcher::SSD_error(const cv::Mat& a, const cv::Mat& b) {
	if (a.size != b.size) {
		throw invalid_argument("image sizes must agree");
	}
	long long error = 0;
	for (int r = 0; r < a.rows; ++r) {
		for (int c = 0; c < a.cols; ++c) {
			Vec3b pa = a.at<Vec3b>(r, c), pb = b.at<Vec3b>(r, c);
			if (pa != Vec3b(0, 0, 0) && pb != Vec3b(0, 0, 0)) {
				for (int i = 0; i < 3; ++i) {
					error += ((int)pa[i] - pb[i])*((int)pa[i] - pb[i]);
				}
			}
		}
	}
	return error / a.total();
}