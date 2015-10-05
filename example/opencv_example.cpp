#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/numcp.h"
#include "../include/numcp_opencv.hpp"

int main(int argc, char const* argv[])
{
	cv::Mat in_img = cv::imread("data/misc/test_rgb.tiff", 1);
	std::cout << in_img << std::endl << std::endl;
	return 0;
}
