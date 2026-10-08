#include <iostream>
#include <format>
#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
using namespace std;

struct crater {
	int x;
	int y;
	float width;
	float height;
};

struct detectedPair {
	vector<crater> vec;
	cv::Mat img;
};

struct confusionMatrix {
	int TP;
	int FP;
	int FN;
};

confusionMatrix evaluate(vector<crater> detectedCraters, vector<crater> labelledCraters) {
	confusionMatrix metrics = { 0, 0, 0 };
	
	vector<bool> labelledDetected(labelledCraters.size(), false);

	for (int i = 0; i < detectedCraters.size(); i++) {
		crater detected = detectedCraters[i];
		int x = detected.x;
		int y = detected.y;
		cout << format("{}|x: {}, y: {}", i, x, y) << endl;

		bool matched = false;
		int j = 0;

		for (int j = 0; j < labelledCraters.size(); j++) {

			if (labelledDetected[j]) {
				continue;
			}

			crater labelled = labelledCraters[j];
			
			float distance = sqrt(pow((detected.x - labelled.x), 2) + pow((detected.y - labelled.y), 2));
			
			const int radius = detected.width/2;
			if (distance < radius) { //height and width are the same, 2*radius of detection area
				matched = true;
				labelledDetected[j] = true;
				metrics.TP++;
				break;
			}
		}
		if (!matched) {
			metrics.FP++;
		}
	}

	metrics.FN = count(labelledDetected.begin(), labelledDetected.end(), false);
	return metrics;
}

cv::Mat preprocessing(cv::Mat image) {
	cv::Mat temp = image;

	//resize
	cv::resize(temp, temp, cv::Size(), 1, 1, cv::INTER_AREA);

	cv::Mat preprocessedImg;
	//may not be necessary with moon images
	cv::cvtColor(temp, preprocessedImg, cv::COLOR_RGB2GRAY);

	//gaussian blur requires defining kernel size with cv::size, dimensions of kernel must be odd
	//cv::GaussianBlur(preprocessedImg, preprocessedImg, cv::Size(15, 15), 5, 5);
	cv::medianBlur(preprocessedImg, preprocessedImg, 7);

	cv::Mat dx, dy;
	//sobel image derivative
	cv::Sobel(preprocessedImg, dx, CV_32F, 1, 0, 3);
	cv::Sobel(preprocessedImg, dy, CV_32F, 0, 1, 3);

	////combine dx and dy into single image
	cv::Mat sobelImage;
	cv::magnitude(dx, dy, sobelImage);
	cv::normalize(sobelImage, preprocessedImg, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	cv::waitKey(0);

	cv::imshow("Preprocessed Image", preprocessedImg);
	return preprocessedImg;
}

detectedPair detect_craters(cv::Mat image, cv::Mat preprocessedImg) {
	cv::Mat detectedImg = image;
	cv::Mat preprocessed = preprocessedImg;

	vector<cv::Vec3f> circles;
	vector<crater> detectedCraters;
	try {
		cv::HoughCircles(preprocessed, circles, cv::HOUGH_GRADIENT, 1.5, 70, 150, 50, 1, 100);
	}
	catch (const cv::Exception& e) {
		std::cerr << "OpenCV error: " << e.what() << '\n';
	}

	for (const cv::Vec3f& circle : circles) {
		cv::Point p1(circle[0] - circle[2], circle[1] - circle[2]);
		cv::Point p2(circle[0] + circle[2], circle[1] + circle[2]);
		cv::rectangle(detectedImg, p1, p2, cv::Scalar(0, 255, 0), 3);
		crater temp = { round(circle[0]), round(circle[1]), circle[2] * 2, circle[2] * 2 };
		detectedCraters.push_back(temp);
	}
	cv::imshow("Detected Image", detectedImg);

	return detectedPair{detectedCraters, detectedImg};
}

vector<crater> label_craters(cv::Mat image, string filename) {
	string temp = "data/labels/" + filename;
	cv::Mat labelledImg = image;
	int length = temp.length();
	temp.replace(length - 3, 3, "txt");
	//cout << format("label file for {} is {}", filename, temp) << endl;

	int width = image.cols;
	int height = image.rows;
	cout << format("img width: {} and height {}\n", width, height);

	vector<crater> craters;

	ifstream file(temp);
	if (!file) {
		cout << format("Tried opening file at: {}", temp);
		cerr << "Error opening file" << endl;
	}
	else {
		string line;
		//read file line by line and break line up into words, convert to int and save to array
		while (getline(file, line)) {
			stringstream ss(line);
			string param;
			vector<float> params;
			while (ss >> param) {
				params.push_back(stof(param));
			}
			crater label = { params[1] * width, params[2] * height, params[3] * width / 2, params[4] * height / 2 };
			cv::Point p1(label.x - label.width, label.y - label.height);
			cv::Point p2(label.x + label.width, label.y + label.height);
			cv::rectangle(labelledImg, p1, p2, cv::Scalar(0, 0, 255), 3);

			craters.push_back(label);
		}
	}
	cv::imshow("Labelled Image", labelledImg);
	return craters;
}

int main() {
	
	string dir = "data/images/";

	confusionMatrix metrics = { 0, 0, 0 };

	int i = 0;
	for (const auto& file : std::filesystem::recursive_directory_iterator(dir)) {
		string filename = file.path().filename().string();
		cout << format("file {}: {}", i, dir + filename);
		i++;
		cv::Mat image = cv::imread(dir+filename);

		cv::Mat preprocessedImg = preprocessing(image);

		detectedPair res = detect_craters(image.clone(), preprocessedImg);
		vector<crater> detected = res.vec;
		cv::Mat detectedImg = res.img;
		//cv::imwrite(format("postimage{}.jpg", i), detectedImg);

		vector<crater> labelled = label_craters(image.clone(), filename);
		confusionMatrix confMatrix =  evaluate(detected, labelled);
		cout << format("TP:{}|FP:{}|FN:{}", confMatrix.TP, confMatrix.FP, confMatrix.FN) << endl;

		metrics.TP += confMatrix.TP;
		metrics.FP += confMatrix.FP;
		metrics.FN += confMatrix.FN;

		//cv::waitKey(0);
	}
	
	cout << format("Final Metrics\nTP:{}|FP:{}|FN:{}", metrics.TP, metrics.FP, metrics.FN);

	return 0;
}