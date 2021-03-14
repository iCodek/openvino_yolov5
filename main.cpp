#include <iostream>
#include <string>
#include <vector>

#include "detector.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char *argv[]) {
	Detector* detector = new Detector;
	string xml_path = "C:/Users/85127/Downloads/yolov5_cpp_openvino-master/yolov5_cpp_openvino-master/demo/res/yolov5s.xml";
	detector->init(xml_path);

	VideoCapture capture;
	capture.open(0);
	Mat src;
	while (1) {
		capture >> src;
		vector<Detector::Object> detected_objects;

		auto start = chrono::high_resolution_clock::now();
		detector->process_frame(src, detected_objects);
		for (int i = 0; i < detected_objects.size(); ++i) {
			int xmin = detected_objects[i].rect.x;
			int ymin = detected_objects[i].rect.y;
			int width = detected_objects[i].rect.width;
			int height = detected_objects[i].rect.height;
			Rect rect(xmin, ymin, width, height);
			cv::rectangle(src, rect, Scalar(255, 0, 0), 1, LINE_8, 0);
			cout << "is: " << detected_objects[i].name  << endl;
			putText(src, detected_objects[i].name,
				cv::Point(xmin, ymin - 10),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(0, 0, 0));
		}
		auto end = chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end - start;
		cout << "use " << diff.count() << " s" << endl;
		putText(src, "" + to_string(diff.count()),
			cv::Point(5, 20),
			cv::FONT_HERSHEY_SIMPLEX,
			0.5,
			cv::Scalar(0, 0, 0));
		imshow("cap", src);
		waitKey(1);
	}

	return 0;
}
