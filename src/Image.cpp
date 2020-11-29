#include "Image.h"
#include "kernel.h"

#include <iostream>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

Image::Image(std::string filename, std::string name, enum image_type type) : name(name), type(type)
{
	std::cout << "Importing Image (" << name << ")... ";

	enum cv::ImreadModes mode;

	switch (type)
	{
		case GRAYSCALE:
			mode = cv::IMREAD_GRAYSCALE;
			break;
		default:
			std::cout << "Only GrayScale supported for now" << std::endl;
			exit(1);
			break;
	}

	opencv_mat = cv::imread(filename, mode);

	if (opencv_mat.empty())
	{
		std::cout << std::endl << "Error: the image couldn't be loaded. Exiting..." << std::endl;
		exit(1);

	} else
	{
		std::cout << " Success" << std::endl;

		size.x = opencv_mat.cols;
		size.y = opencv_mat.rows;
		cudaMallocManaged(&dev_ptr, size.x * size.y * sizeof(unsigned char));

		// Copy from data from the OpenCV matrix to a pointer in the Unified Memory
		for(int j = 0; j < size.y; j++)
			for(int i = 0; i < size.x; i++)
				dev_ptr[i + j * size.x] = opencv_mat.at<unsigned char>(j, i);
	}
}

Image::~Image()
{
	cudaFree(dev_ptr);
}

void Image::edge_detection()
{
	apply_sobel_filter();
}

void Image::binomial_filter(const int n_pass)
{
	apply_binomial_filter(n_pass);
}

void Image::save(std::string filename)
{
	// Copy the data back to the OpenCV matrix
	for(int j = 0; j < size.y; j++)
		for(int i = 0; i < size.x; i++)
			opencv_mat.at<unsigned char>(j, i) = dev_ptr[i + j * size.x];

	cv::imwrite(filename, opencv_mat);
}

void Image::diplay()
{
	// Copy the data back to the OpenCV matrix
	for(int j = 0; j < size.y; j++)
		for(int i = 0; i < size.x; i++)
			opencv_mat.at<unsigned char>(j, i) = dev_ptr[i + j * size.x];

	// Display the image
	cv::namedWindow(name);
	cv::imshow(name, opencv_mat);
	cv::waitKey(0); // Press Enter to close the image
	cv::destroyWindow(name);
}
