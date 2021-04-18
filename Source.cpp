#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <numeric>
#include <vector>
#include<opencv2/xfeatures2d.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include<cmath>
#include<cassert>

using namespace std;
using namespace cv;

#define NUM_OF_PAIR 4
#define CONFIDENCE 0.99
#define INLINER_RATIO 0.5
#define RANSAC_THRESHOLD 4.0

struct panorama_matches {
	Mat homography;
	vector<int> indices;
};

double* GenerateWeightMap(const int& height, const int& width)
{
	double max = 0.0;

	// build buffer
	double* buffer = new double[height*width];

	// compute weight for each location in the buffer
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			// find the minimum number of pixels (distance) to an edge of the image
			double dist = std::min(std::min(r, height - r), std::min(c, width - c)) + 1;
			buffer[(r*width) + c] = dist;

			if (dist > max)
			{
				max = dist;
			}
		}
	}

	// normalize
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			buffer[(r*width) + c] /= max;
		}
	}

	return buffer;
}

bool BilinearInterpolation(Mat image, double x, double y, double rgb[3])
{
	// initialize rgb to black
	rgb[0] = 0.0;
	rgb[1] = 0.0;
	rgb[2] = 0.0;

	// compute base pixel
	int baseX = (int)x;
	int baseY = (int)y;

	// check if pixels in range
	if (x >= 0 && (x + 1) < image.cols && y >= 0 && (y + 1) < image.rows) {

		// compute weight values
		double a = x - baseX; //x->columns 
		double b = y - baseY; //y->rows

		if (a >= image.rows) {
			cout << " rows exceeded" << endl;
		}
		if (b >= image.cols) {
			cout << " cols exceeded" << endl;

		}

		// find pixels
		Vec3b pixelXY = image.at<Vec3b>(baseY, baseX);
		Vec3b pixelX1Y = image.at<Vec3b>(baseY + 1, baseX);
		Vec3b pixelXY1 = image.at<Vec3b>(baseY, baseX + 1);
		Vec3b pixelX1Y1 = image.at<Vec3b>(baseY + 1, baseX + 1);

		// compute interpolated pixel
		// f (x + a, y + b) = (1 - a)(1 - b) f (x, y) + a(1 - b) f (x + 1, y) + (1 - a)b f (x,y + 1) + ab f (x + 1, y + 1)
		rgb[0] = ((1 - a)*(1 - b)*(pixelXY[2])) + (a*(1 - b)*(pixelX1Y[2])) + ((1 - a)*b*(pixelXY1[2])) + (a*b*(pixelX1Y1[2]));//--->2
		rgb[1] = ((1 - a)*(1 - b)*(pixelXY[1])) + (a*(1 - b)*(pixelX1Y[1])) + ((1 - a)*b*(pixelXY1[1])) + (a*b*(pixelX1Y1[1]));//--->1
		rgb[2] = ((1 - a)*(1 - b)*(pixelXY[0])) + (a*(1 - b)*(pixelX1Y)[0]) + ((1 - a)*b*(pixelXY1[0])) + (a*b*(pixelX1Y1[0])); //--->0


		// cap rgb values
		rgb[0] = std::max(rgb[0], 0.0);
		rgb[0] = std::min(rgb[0], 255.0);
		rgb[1] = std::max(rgb[1], 0.0);
		rgb[1] = std::min(rgb[1], 255.0);
		rgb[2] = std::max(rgb[2], 0.0);
		rgb[2] = std::min(rgb[2], 255.0);

		return true;
	}

	return false;
}

bool BilinearInterpolation2(Mat image, double x, double y, double rgb)
{
	// initialize rgb to black
	rgb = 0.0;

	// compute base pixel
	int baseX = (int)x;
	int baseY = (int)y;

	// check if pixels in range
	if (x >= 0 && (x + 1) < image.cols && y >= 0 && (y + 1) < image.rows) {

		// compute weight values
		double a = x - baseX; //x->columns 
		double b = y - baseY; //y->rows

		if (a >= image.rows) {
			cout << " rows exceeded" << endl;
		}
		if (b >= image.cols) {
			cout << " cols exceeded" << endl;

		}


		// find pixels
		float pixelXY = (float)image.at<uchar>(baseY, baseX);
		float pixelX1Y = (float)image.at<uchar>(baseY + 1, baseX);
		double pixelXY1 = (float)image.at<uchar>(baseY, baseX + 1);
		double pixelX1Y1 = (float)image.at<uchar>(baseY + 1, baseX + 1);

		// compute interpolated pixel
		// f (x + a, y + b) = (1 - a)(1 - b) f (x, y) + a(1 - b) f (x + 1, y) + (1 - a)b f (x,y + 1) + ab f (x + 1, y + 1)
		rgb = ((1 - a)*(1 - b)*(pixelXY)) + (a*(1 - b)*(pixelX1Y)) + ((1 - a)*b*(pixelXY1)) + (a*b*(pixelX1Y1));//--->2



		// cap rgb values
		rgb = std::max(rgb, 0.0);
		rgb = std::min(rgb, 255.0);


		return true;
	}

	return false;
}


void Project(double x, double y, double &x2, double &y2, Mat H)
{
	// compute resulting matrix [H][xy1] = [uvw]
	float w = (float)(H.at<double>(2, 0) * x + H.at<double>(2, 1)*y + H.at<double>(2, 2));
	x2 = (float)((H.at<double>(0, 0)*x + H.at<double>(0, 1)*y + H.at<double>(0, 2)) / w);

	y2 = (float)((H.at<double>(1, 0)*x + H.at<double>(1, 1)*y + H.at<double>(1, 2)) / w);

}

void Stitch(Mat image1, Mat image2, Mat hom, Mat homInv, Mat &stitchedImage)
{


	/*imshow("image 1", image1);
	imshow("image 2", image2);

	cout << " the dimensions(image_1) " << image1.cols << " X " << image1.rows << endl;
	cout << " the dimensions(image_2) " << image2.cols << " X " << image2.rows << endl;*/
	// width and height of stitched image
	int ws = 0;
	int hs = 0;

	// project the four corners of image 2 onto image 1
	double image2TopLeft[2] = { 0, 0 };
	double image2TopRight[2] = { image2.cols - 1, 0 };
	double image2BottomLeft[2] = { 0, image2.rows - 1 };
	double image2BottomRight[2] = { image2.cols - 1, image2.rows - 1 };

	Project(image2TopLeft[0], image2TopLeft[1], image2TopLeft[0], image2TopLeft[1], homInv);
	Project(image2TopRight[0], image2TopRight[1], image2TopRight[0], image2TopRight[1], homInv);
	Project(image2BottomLeft[0], image2BottomLeft[1], image2BottomLeft[0], image2BottomLeft[1], homInv);
	Project(image2BottomRight[0], image2BottomRight[1], image2BottomRight[0], image2BottomRight[1], homInv);

	// compute the size of stitched image, minimum top-left position and maximum bottom-right position
	int top = std::min(0, (int)std::min(image2TopLeft[1], image2TopRight[1]));
	int left = std::min(0, (int)std::min(image2TopLeft[0], image2BottomLeft[0]));
	int bottom = std::max(image1.rows, (int)(std::max(image2BottomRight[1], image2BottomLeft[1]) + 1.0));
	int right = std::max(image1.cols, (int)(std::max(image2BottomRight[0], image2TopRight[0]) + 1.0));

	ws = right - left + 1;
	hs = bottom - top + 1;

	// generate weight maps for images
	double* image1Weights = GenerateWeightMap(image1.rows, image1.cols);
	double* image2Weights = GenerateWeightMap(image2.rows, image2.cols);

	// initialize stiched image
	stitchedImage = Mat(hs, ws, image1.type(), cv::Scalar::all(0));

	//cout << " the dimensions(panorama) " << ws << " X " << hs << endl;



	// copy image1 into stitched image at the proper location
	for (int r = 0; r < image1.rows; r++)
	{
		for (int c = 0; c < image1.cols; c++)
		{
			stitchedImage.at<Vec3b>(r + abs(top), c + abs(left)) = image1.at<Vec3b>(r, c);

		}
	}


	// for each pixel in stitched image, 
	for (int r = top; r < bottom; r++)
	{
		//cout << "r->" << r;
		for (int c = left; c < right; c++)
		{
			double x2 = 0.0;
			double y2 = 0.0;

			// project point onto image2
			Project(c, r, x2, y2, hom);

			// interpolate image2 pixel
			double rgb[3];
			//cout << "(" << r << "," << c << ")";

			//cout << "c->"<<c;
			bool in = BilinearInterpolation(image2, x2, y2, rgb);

			if (in == true) {
				//cout << "c->" << c;

				// stiched image row and column
				int sRow = r + std::abs(top);
				int sCol = c + std::abs(left);

				// check if overlap with image1 pixel
				// combine pixels based on weight map
				if (sRow >= std::abs(top) && sRow < abs(top) + image1.rows && sCol >= abs(left) && sCol < abs(left) + image1.cols) {
					// verify pixel is not part of black borders
					if (sCol < stitchedImage.rows && stitchedImage.at<Vec3b>(sRow, sCol)[2] != 0 && stitchedImage.at<Vec3b>(sRow, sCol)[1] != 0 && stitchedImage.at<Vec3b>(sRow, sCol)[0] != 0) {
						// compute image 1 pixel
						int i1Row = sRow - std::abs(top);
						int i1Col = sCol - std::abs(left);

						// find image weights
						double i1Weight = image1Weights[i1Row*image1.cols + i1Col];
						double i2Weight = image2Weights[((int)y2)*image2.cols + (int)x2];

						// normalize weights
						double totalWeight = i1Weight + i2Weight;
						i1Weight /= totalWeight;
						i2Weight /= totalWeight;

						// compute new rgb values
						double image2rgb[3] = { rgb[0], rgb[1], rgb[2] };

						rgb[0] = i1Weight * stitchedImage.at<Vec3b>(sRow, sCol)[2] + i2Weight * image2rgb[0];
						rgb[1] = i1Weight * stitchedImage.at<Vec3b>(sRow, sCol)[1] + i2Weight * image2rgb[1];
						rgb[2] = i1Weight * stitchedImage.at<Vec3b>(sRow, sCol)[0] + i2Weight * image2rgb[2];




					}
				}/**/




				// AVERAGE PIXELS
				if (stitchedImage.at<Vec3b>(sRow, sCol)[2] != 0 && stitchedImage.at<Vec3b>(sRow, sCol)[1] != 0 && stitchedImage.at<Vec3b>(sRow, sCol)[0] != 0)
				{
					// average with existing pixel
					rgb[0] = (rgb[0] + stitchedImage.at<Vec3b>(sRow, sCol)[2]) / 2;//2
					rgb[1] = (rgb[1] + stitchedImage.at<Vec3b>(sRow, sCol)[1]) / 2;//1
					rgb[2] = (rgb[2] + stitchedImage.at<Vec3b>(sRow, sCol)[0]) / 2;//0
				}/**/

				// add image2 pixel to stitched image
				stitchedImage.at<Vec3b>(sRow, sCol)[0] = rgb[2];
				stitchedImage.at<Vec3b>(sRow, sCol)[1] = rgb[1];
				stitchedImage.at<Vec3b>(sRow, sCol)[2] = rgb[0];






			}


		}
	}

	delete[] image1Weights;
	delete[] image2Weights;
}

void Stitch2(Mat image1, Mat image2, Mat hom, Mat homInv, Mat &stitchedImage)
{

	/*imshow("image 1", image1);
	imshow("image 2", image2);

	cout << " the dimensions(image_1) " << image1.cols << " X " << image1.rows << endl;
	cout << " the dimensions(image_2) " << image2.cols << " X " << image2.rows << endl;*/
	// width and height of stitched image
	int ws = 0;
	int hs = 0;

	// project the four corners of image 2 onto image 1
	double image2TopLeft[2] = { 0, 0 };
	double image2TopRight[2] = { image2.cols - 1, 0 };
	double image2BottomLeft[2] = { 0, image2.rows - 1 };
	double image2BottomRight[2] = { image2.cols - 1, image2.rows - 1 };

	Project(image2TopLeft[0], image2TopLeft[1], image2TopLeft[0], image2TopLeft[1], homInv);
	Project(image2TopRight[0], image2TopRight[1], image2TopRight[0], image2TopRight[1], homInv);
	Project(image2BottomLeft[0], image2BottomLeft[1], image2BottomLeft[0], image2BottomLeft[1], homInv);
	Project(image2BottomRight[0], image2BottomRight[1], image2BottomRight[0], image2BottomRight[1], homInv);

	// compute the size of stitched image, minimum top-left position and maximum bottom-right position
	int top = std::min(0, (int)std::min(image2TopLeft[1], image2TopRight[1]));
	int left = std::min(0, (int)std::min(image2TopLeft[0], image2BottomLeft[0]));
	int bottom = std::max(image1.rows, (int)(std::max(image2BottomRight[1], image2BottomLeft[1]) + 1.0));
	int right = std::max(image1.cols, (int)(std::max(image2BottomRight[0], image2TopRight[0]) + 1.0));

	ws = right - left + 1;
	hs = bottom - top + 1;

	// generate weight maps for images
	double* image1Weights = GenerateWeightMap(image1.rows, image1.cols);
	double* image2Weights = GenerateWeightMap(image2.rows, image2.cols);

	// initialize stiched image
	stitchedImage = Mat(hs, ws, image1.type(), cv::Scalar::all(0));

	//cout << " the dimensions(panorama) " << ws << " X " << hs << endl;



	// copy image1 into stitched image at the proper location
	for (int r = 0; r < image1.rows; r++)
	{
		for (int c = 0; c < image1.cols; c++)
		{
			stitchedImage.at<uchar>(r + abs(top), c + abs(left)) = image1.at<uchar>(r, c);

		}
	}


	// for each pixel in stitched image, 
	for (int r = top; r < bottom; r++)
	{
		//cout << "r->" << r;
		for (int c = left; c < right; c++)
		{
			double x2 = 0.0;
			double y2 = 0.0;

			// project point onto image2
			Project(c, r, x2, y2, hom);

			// interpolate image2 pixel
			double rgb = 0.0;
			//cout << "(" << r << "," << c << ")";

			//cout << "c->"<<c;
			bool in = BilinearInterpolation2(image2, x2, y2, rgb);

			if (in == true) {
				//cout << "c->" << c;

				// stiched image row and column
				int sRow = r + std::abs(top);
				int sCol = c + std::abs(left);

				// check if overlap with image1 pixel
				// combine pixels based on weight map
				if (sRow >= std::abs(top) && sRow < abs(top) + image1.rows && sCol >= abs(left) && sCol < abs(left) + image1.cols) {
					// verify pixel is not part of black borders
					if (sCol < stitchedImage.rows && (float)stitchedImage.at<uchar>(sRow, sCol) != 0) {
						// compute image 1 pixel
						int i1Row = sRow - std::abs(top);
						int i1Col = sCol - std::abs(left);

						// find image weights
						double i1Weight = image1Weights[i1Row*image1.cols + i1Col];
						double i2Weight = image2Weights[((int)y2)*image2.cols + (int)x2];

						// normalize weights
						double totalWeight = i1Weight + i2Weight;
						i1Weight /= totalWeight;
						i2Weight /= totalWeight;

						// compute new rgb values
						double image2rgb = rgb;

						rgb = i1Weight * (float)stitchedImage.at<uchar>(sRow, sCol) + i2Weight * image2rgb;





					}
				}




				// AVERAGE PIXELS
				if ((float)stitchedImage.at<uchar>(sRow, sCol) != 0)
				{
					// average with existing pixel
					rgb = (rgb + (float)stitchedImage.at<uchar>(sRow, sCol)) / 2;
				}

				// add image2 pixel to stitched image
				stitchedImage.at<uchar>(sRow, sCol) = rgb;

			}

		}
	}

	delete[] image1Weights;
	delete[] image2Weights;
}


int numberOfIterations(float p, float w, int num) {
	return ceil(log(1 - p) / log(1 - pow(w, num)));
}

vector<int> inlier_count(Mat H, vector<Point2f>& obj, vector<Point2f>& scene) {

	vector<int>inlier_indices;


	for (int i = 0; i < obj.size(); i++) {


		float real_x = scene[i].x;
		float real_y = scene[i].y;

		float w = (float)(H.at<double>(2, 0) * obj[i].x + H.at<double>(2, 1)*obj[i].y + H.at<double>(2, 2));

		float x = (float)((H.at<double>(0, 0)*obj[i].x + H.at<double>(0, 1)*obj[i].y + H.at<double>(0, 2)) / w);

		float y = (float)((H.at<double>(1, 0)*obj[i].x + H.at<double>(1, 1)*obj[i].y + H.at<double>(1, 2)) / w);


		//Euclidean Distance
		float distance = sqrt((x - real_x) * (x - real_x) + (y - real_y) * (y - real_y));

		if (distance < RANSAC_THRESHOLD) {//RANSAC_THRESHOLD = 4.0f
			inlier_indices.push_back(i);
		}
	}

	return inlier_indices;

}

panorama_matches ransac(vector<Point2f>& obj, vector<Point2f>& scene) {

	int iterations = (int)numberOfIterations(CONFIDENCE, INLINER_RATIO, NUM_OF_PAIR);
	vector<int> MAX_inliers;// CONTAINS THE INDICES OF THE SELECTED COORDINATES THAT MOST OF THE POINTS COINCIDE WITH RESPECT TO THE HOMOGRAPHY CALCULATED
	int indices[4];
	//std::cout << "Number of Iterations : " << iterations << endl;

	while (iterations--) {

		vector<int> current_inliers;// ->Index number of the current inliers

		vector<Point2f>random_set_obj(4); //4 random pairs of cooridinates that are matches  
		vector<Point2f>random_set_scene(4);// between the 2 images

		int index_1 = rand() % obj.size();
		random_set_obj[0] = obj[index_1];
		random_set_scene[0] = scene[index_1];
		//cout << obj[index_1] << " , " << scene[index_1]<<endl;

		int index_2 = rand() % obj.size();
		while (index_1 == index_2) {
			index_2 = rand() % obj.size();
		}
		random_set_obj[1] = obj[index_2];
		random_set_scene[1] = scene[index_2];
		//cout << obj[index_2] << " , " << scene[index_2] << endl;


		int index_3 = rand() % obj.size();
		while (index_3 == index_1 && index_3 == index_2) {
			index_3 = rand() % obj.size();
		}
		random_set_obj[2] = obj[index_3];
		random_set_scene[2] = scene[index_3];
		//cout << obj[index_3] << " , " << scene[index_3] << endl;


		int index_4 = rand() % obj.size();
		while (index_4 == index_1 && index_4 == index_2 && index_4 == index_3) {
			index_4 = rand() % obj.size();
		}
		random_set_obj[3] = obj[index_4];
		random_set_scene[3] = scene[index_4];
		//cout << obj[index_4] << " , " << scene[index_4] << endl;





		Mat H = findHomography(random_set_obj, random_set_scene, 0); // Finds the Homography of the 4 pairs of coordinates

		// Finds the index of the current inliers
		current_inliers = inlier_count(H, obj, scene);


		if (current_inliers.size() > MAX_inliers.size()) {
			MAX_inliers = current_inliers;
			indices[0] = index_1;
			indices[1] = index_2;
			indices[2] = index_3;
			indices[3] = index_4;

		}

	}
	//cout << "MAX INLIERS DETECTED : " << MAX_inliers.size() << endl;

	vector<Point2f> final_obj;
	final_obj.push_back(obj[indices[0]]);
	final_obj.push_back(obj[indices[1]]);
	final_obj.push_back(obj[indices[2]]);
	final_obj.push_back(obj[indices[3]]);

	vector<Point2f> final_scene;
	final_scene.push_back(scene[indices[0]]);
	final_scene.push_back(scene[indices[1]]);
	final_scene.push_back(scene[indices[2]]);
	final_scene.push_back(scene[indices[3]]);

	Mat homography = findHomography(final_obj, final_scene, 0);
	//cout << "\nHomography selected from RANSAC implemented by me\n " << findHomography(final_obj, final_scene) << endl;

	//cout << "\nHomography from the inbuilt ransac function\n" << findHomography(obj, scene, RANSAC) << '\n' << endl;

	panorama_matches result;
	result.homography = homography;
	result.indices = MAX_inliers;

	return result;

}

int number_of_matches(Mat img_1, Mat img_2) {
	// Detector and Descriptor
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	Mat descriptor_1, descriptor_2;
	vector<KeyPoint> keypoints_1, keypoints_2;

	Mat grey1, grey2;

	cvtColor(img_1, grey1, cv::COLOR_BGR2GRAY);
	cvtColor(img_2, grey2, cv::COLOR_BGR2GRAY);


	detector->detect(grey1, keypoints_1);
	detector->compute(grey1, keypoints_1, descriptor_1);

	detector->detect(grey2, keypoints_2);
	detector->compute(grey2, keypoints_2, descriptor_2);

	//Matcher
	cv::FlannBasedMatcher   matcher;
	vector<cv::DMatch> matches;
	matcher.match(descriptor_1, descriptor_2, matches);

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptor_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	int number_of_matches = 0;
	for (int i = 0; i < descriptor_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			number_of_matches++;
		}
	}

	return number_of_matches;
}

Mat run(Mat img_1, Mat img_2) {

	// Detector and Descriptor
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	Mat descriptor_1, descriptor_2;
	vector<KeyPoint> keypoints_1, keypoints_2;

	Mat grey1, grey2;

	cvtColor(img_1, grey1, cv::COLOR_BGR2GRAY);
	cvtColor(img_2, grey2, cv::COLOR_BGR2GRAY);

	detector->detect(grey1, keypoints_1);
	detector->compute(grey1, keypoints_1, descriptor_1);

	detector->detect(grey2, keypoints_2);
	detector->compute(grey2, keypoints_2, descriptor_2);

	//Matcher
	cv::FlannBasedMatcher   matcher;
	vector<cv::DMatch> matches;
	matcher.match(descriptor_1, descriptor_2, matches);

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptor_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	vector< DMatch > good_matches;

	for (int i = 0; i < descriptor_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	vector<Point2f> obj, scene;
	const unsigned nMatches = good_matches.size();
	for (unsigned i = 0; i < nMatches; i++) {
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);//image_1
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);//image_2
	}

	//Mat img_matches;
	//drawMatches(img_1, keypoints_1, img_2, keypoints_2,good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	//imshow("Good Matches", img_matches);

	// Run RANSAC on obj and scene
	panorama_matches blend = ransac(obj, scene);

	vector<Point2f>new_obj(blend.indices.size()), new_scene(blend.indices.size());
	for (int i = 0; i < blend.indices.size(); i++) {
		new_obj[i] = obj[blend.indices[i]];
		new_scene[i] = scene[blend.indices[i]];


	}


	Mat HOMOGRAPHY = findHomography(new_obj, new_scene, 0);// Homography calculated with all the inliers.
	Mat inv_HOMOGRAPHY = HOMOGRAPHY.inv();


	Mat pano;
	Stitch(img_1, img_2, HOMOGRAPHY, inv_HOMOGRAPHY, pano);
	return pano;




}

Mat run2(Mat img_1, Mat img_2) {

	// Detector and Descriptor
	Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
	Mat descriptor_1, descriptor_2;
	vector<KeyPoint> keypoints_1, keypoints_2;

	Mat grey1 = img_1, grey2 = img_2;

	//cvtColor(img_1, grey1, cv::COLOR_BGR2GRAY);
	//cvtColor(img_2, grey2, cv::COLOR_BGR2GRAY);

	detector->detect(grey1, keypoints_1);
	detector->compute(grey1, keypoints_1, descriptor_1);

	detector->detect(grey2, keypoints_2);
	detector->compute(grey2, keypoints_2, descriptor_2);

	//Matcher
	cv::FlannBasedMatcher   matcher;
	vector<cv::DMatch> matches;
	matcher.match(descriptor_1, descriptor_2, matches);
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imwrite("2.png", img_matches);

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptor_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	vector< DMatch > good_matches;

	for (int i = 0; i < descriptor_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}
	Mat ing_goodMatches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, ing_goodMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imwrite("3.png", ing_goodMatches);

	vector<Point2f> obj, scene;
	const unsigned nMatches = good_matches.size();
	for (unsigned i = 0; i < nMatches; i++) {
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);//image_1
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);//image_2
	}

	// Run RANSAC on obj and scene
	panorama_matches blend = ransac(obj, scene);

	vector<Point2f>new_obj(blend.indices.size()), new_scene(blend.indices.size());
	cout << blend.indices.size() << endl;
	for (int i = 0; i < blend.indices.size(); i++) {
		new_obj[i] = obj[blend.indices[i]];
		new_scene[i] = scene[blend.indices[i]];


	}


	Mat HOMOGRAPHY = findHomography(new_obj, new_scene, 0);// Homography calculated with all the inliers.
	Mat inv_HOMOGRAPHY = HOMOGRAPHY.inv();

	Mat pano;
	Stitch2(img_1, img_2, HOMOGRAPHY, inv_HOMOGRAPHY, pano);
	return pano;




}

int main(int argc, char *argv[])
{
	cout << "Running Source code" << endl;

	Mat img_1 = imread("Rainier1.png");//obj
	Mat img_2 = imread("Rainier2.png");//scene
	Mat img_3 = imread("Rainier3.png");
	Mat img_4 = imread("Rainier4.png");
	Mat img_5 = imread("Rainier5.png");
	Mat img_6 = imread("Rainier6.png");/**/

	/*Mat pano = run(img_1, img_6);
	pano = run(pano, img_4);
	pano = run(pano, img_5);
	pano = run(pano, img_2);
	pano = run(pano, img_3);
	imshow("panorama", pano);*/

	/*vector<Mat>images;
	images.push_back(img_1);
	images.push_back(img_2);
	images.push_back(img_3);
	images.push_back(img_4);
	images.push_back(img_5);
	images.push_back(img_6);


	for (int i = 0; i < images.size(); i++) {
		cout << "Image number  " << i + 1<<" "<<endl;
		int sum = 0;
		for (int j = 0; j < images.size(); j++) {
			if (i != j) {
				sum += number_of_matches(images[i], images[j]);

				//cout << "with image number " << j + 1 << " " << number_of_matches(images[i], images[j]) << endl;
			}


		}
		cout<<" total matches: ="<<sum;
		cout << endl;
	}
	cin.get();*/

	//**************Code to generate "2.png" , "3.png" and "4.png" **************************
	/*run2(img_1, img_2);

	imwrite("4.png",run(img_1,img_2));*/

	//****************************************************************************


	//**************Best Order (Rainier Pictures) *********************************
	//Best Order- 1-6-4-3-5-2
	/*Mat pano = run(img_1, img_6);
	pano = run(pano, img_4);
	pano = run(pano, img_3);
	pano = run(pano, img_5);
	pano = run(pano, img_2);
	imwrite("stitched_Rainier.png", pano);
	imshow("Stitched Image", pano);
	*/
	//*****************************************************************************


	//************** Home Pictures *************************************************
	//Best Order- 6-5-7
	Mat window1_scaled, window1;
	window1 = imread("lab6.jpg");
	resize(window1, window1_scaled, Size(), 0.18, 0.18);
	//cout<<"(" << window1_scaled.rows<<" , " << window1_scaled.cols<<")" << endl;
	//imshow("scaled 1 ", window1_scaled);

	Mat window2_scaled, window2;
	window2 = imread("lab5.jpg");
	resize(window2, window2_scaled, Size(), 0.18, 0.18);
	//cout << "(" << window2_scaled.rows << " , " << window2_scaled.cols << ")" << endl;
	//imshow("scaled 2 ", window2_scaled);

	Mat pano = run(window1_scaled, window2_scaled);

	Mat window3_scaled, window3;
	window3 = imread("lab7.jpg");
	resize(window3, window3_scaled, Size(), 0.18, 0.18);
	//cout << "(" << window3_scaled.rows << " , " << window3_scaled.cols << ")" << endl;
	//imshow("scaled 3 ", window3_scaled);
	pano = run(pano, window3_scaled);
	
	//Mat window4_scaled, window4;
	//window4 = imread("lab4.jpg");
	//resize(window4, window4_scaled, Size(), 0.18, 0.18);
	//cout << "(" << window4_scaled.rows << " , " << window4_scaled.cols << ")" << endl;
	//imshow("scaled 4 ", window4_scaled);
	//pano = run(pano, window4_scaled);
/*
	/*imwrite("executed_image.png", pano);*/

	imshow("Stitched_Image.png", pano);/**/

//*****************************************************************************


//************* Hanging Pictures **********************************************
/*Mat hang_1 = imread("Hanging1.png");
Mat hang_2 = imread("Hanging2.png");

Mat pano = run(hang_1, hang_2);
imshow("Hanging", pano);*/
//*****************************************************************************


//************ ND **************************************************************
/*Mat ND1 = imread("ND1.png", IMREAD_GRAYSCALE);
Mat ND2 = imread("ND2.png", IMREAD_GRAYSCALE);
Mat pano = run2(ND1, ND2);
imshow("ND", pano);*/
//*****************************************************************************


/*Mat window1_scaled, window1;
window1 = imread("lab1.jpg");
resize(window1, window1_scaled, Size(),0.18, 0.18);
//cout<<"(" << window1_scaled.rows<<" , " << window1_scaled.cols<<")" << endl;
//imshow("scaled 1 ", window1_scaled);

Mat window2_scaled, window2;
window2 = imread("lab2.jpg");
resize(window2, window2_scaled, Size(), 0.18, 0.18);
//cout << "(" << window2_scaled.rows << " , " << window2_scaled.cols << ")" << endl;
//imshow("scaled 2 ", window2_scaled);

Mat pano = run(window1_scaled, window2_scaled);

Mat window3_scaled, window3;
window3 = imread("lab3.jpg");
resize(window3, window3_scaled, Size(), 0.18, 0.18);
//cout << "(" << window3_scaled.rows << " , " << window3_scaled.cols << ")" << endl;
//imshow("scaled 3 ", window3_scaled);
pano = run(pano, window3_scaled);

//Mat window4_scaled, window4;
//window4 = imread("lab4.jpg");
//resize(window4, window4_scaled, Size(), 0.18, 0.18);
//cout << "(" << window4_scaled.rows << " , " << window4_scaled.cols << ")" << endl;
//imshow("scaled 4 ", window4_scaled);
//pano = run(pano, window4_scaled);*/

/*/*imwrite("executed_image.png", pano);*/

/*imshow("Stitched_Image.png", pano);*/






//cin.get();



	waitKey();

	return 0;
}