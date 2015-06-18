#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

int num_images = 14;
vector<string> img_names(num_images);
auto img_names_generator = [] (int idx) {return "images/IMG" + to_string(idx + 1) + ".jpg"; };
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
float conf_thresh = 1.f;
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;

int main() {
	for (int i = 0; i < num_images; ++i) {
		img_names[i] = img_names_generator(i);
	}
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_compose_scale_set = false;
	
	Ptr<FeaturesFinder> finder{ makePtr<OrbFeaturesFinder>() };

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i) {
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (i == 0) {
			work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
		}
		seam_work_aspect = seam_scale / work_scale;
		cv::resize(full_img, img, Size(), work_scale, work_scale);

		(*finder)(img, features[i]);
		features[i].img_idx = i;

		resize(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(try_cuda, match_conf);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();

	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<string> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i) {
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;
	num_images = images.size();

	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	if (!estimator(features, pairwise_matches, cameras)) {
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (size_t i = 0; i < cameras.size(); ++i) {
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	Ptr<detail::BundleAdjusterBase> adjuster{ makePtr<detail::BundleAdjusterRay>() };
	adjuster->setConfThresh(conf_thresh);
	if (!(*adjuster)(features, pairwise_matches, cameras)) {
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}

	// fokusztavok medianja
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i) {
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct) {
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	vector<Point> corners(num_images);
	vector<Size> sizes(num_images);
	Ptr<WarperCreator> warper_creator{ makePtr<cv::CylindricalWarper>() };
	Ptr<RotationWarper> warper;

	/*vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	
	vector<UMat> masks(num_images);

	for (int i = 0; i < num_images; ++i) {
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i) {
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder{ makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR) };
	seam_finder->find(images_warped_f, corners, masks_warped);

	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();*/

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	double compose_work_aspect = 1;

	for (int img_idx = 0; img_idx < num_images; ++img_idx) {

		full_img = imread(img_names[img_idx]);
		if (!is_compose_scale_set) {
			is_compose_scale_set = true;

			compose_work_aspect = compose_scale / work_scale;

			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			for (int i = 0; i < num_images; ++i) {
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1) {
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img, img, Size(), compose_scale, compose_scale);
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		if (!blender) {
			blender = Blender::createDefault(blend_type, try_cuda);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);
			else if (blend_type == Blender::MULTI_BAND) {
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			}
			else if (blend_type == Blender::FEATHER) {
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				fb->setSharpness(1.f / blend_width);
			}
			blender->prepare(corners, sizes);
		}

		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	}


	Mat result, result_mask;
	blender->blend(result, result_mask);

	imwrite("panorama.jpg", result);

	return 0;
}