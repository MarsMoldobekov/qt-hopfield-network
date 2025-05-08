#include <opencv4/opencv2/opencv.hpp>

#include "image_processor.h"

std::vector<NeuronNet::State> ImageProcessor::loadImage(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Couldn't load image: " + path);
    }
    return preprocessImage(image);
}

std::vector<NeuronNet::State> ImageProcessor::preprocessImage(const cv::Mat &image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(100, 100));

    cv::Mat blurred;
    cv::GaussianBlur(resized, blurred, cv::Size(3, 3), 0);

    cv::Mat binary;
    cv::threshold(blurred, binary, 128, 255, cv::THRESH_BINARY);

    binary = binary.reshape(1, 1);
    std::vector<NeuronNet::State> states;
    states.reserve(100*100);

    for (int i = 0; i < binary.cols; ++i) {
        states.push_back(binary.at<uchar>(0, i) < 128 ? NeuronNet::State::Upper : NeuronNet::State::Lower);
    }

    return states;
}

void ImageProcessor::saveImage(const std::vector<NeuronNet::State> &states, const std::string &path, int width, int height) {
    cv::Mat image(height, width, CV_8UC1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            image.at<uchar>(y, x) = states[idx] == NeuronNet::State::Upper ? 0 : 255;
        }
    }

    cv::imwrite(path, image);
}
