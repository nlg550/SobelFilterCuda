#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <string>
#include <cuda.h>
#include <vector_types.h> // CUDA built-in vector types
#include <opencv4/opencv2/core.hpp> // Using OpenCV to read JPEG images and display on the screen

enum image_type {
	GRAYSCALE, COLOR
};

class Image {
private:
	cv::Mat opencv_mat; 	// OpenCV Matrix
	std::string name;
	enum image_type type;
	int2 size;				// Image size (x, y)
	unsigned char* dev_ptr; // The Sobel Filter in applied in Gray Scale

	// Wrapper for CUDA kernels
	void apply_sobel_filter();
	void apply_binomial_filter(const int n_pass);

public:
	// Import image file
	Image(std::string filename, std::string name, enum image_type type);
	virtual ~Image();

	void edge_detection(); 					// Apply the Sobel Filter
	void binomial_filter(const int n_pass); // Apply a binomial filter n_pass times
	void diplay(); 							// Display
	void save(std::string filename);		// Save image
};

#endif /* SRC_IMAGE_H_ */
