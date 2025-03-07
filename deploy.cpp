#include <iostream>
#include <opencv2/opencv.hpp>
#include "tm/interface.h"

using namespace cv;

int main()
{
    Mat image = imread("84.jpg", IMREAD_COLOR);

    cv::Mat img;
    img = image;
    img.convertTo(img, CV_32F);

    // Normalize pixel values
    cv::Mat normalized_image;
    img.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

    // Resize image
    resize(normalized_image, normalized_image, cv::Size(640, 640));

    cv::Scalar mean(0, 0, 0);
    cv::Scalar std(0.5, 0.5, 0.5);

    normalized_image = (normalized_image - mean) / std;
    
    //224 for resnet 640 for yolov8
    cv::Mat inputBlob = cv::dnn::blobFromImage(normalized_image, 1.0, cv::Size(), cv::Scalar(), true, false);

    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
    net.setInput(inputBlob);

    cv::Mat result = net.forward();
    cv::Point classIdPoint;
}
