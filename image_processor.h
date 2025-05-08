#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <string>
#include <opencv2/core/mat.hpp>

#include "neural_network.h"

class ImageProcessor {
public:
    static std::vector<NeuronNet::State> loadImage(const std::string &path);

    static void saveImage(const std::vector<NeuronNet::State> &states,
                          const std::string &path,
                          int width,
                          int height);

    static std::vector<NeuronNet::State> preprocessImage(const cv::Mat &image);
};

#endif // IMAGE_PROCESSOR_H
