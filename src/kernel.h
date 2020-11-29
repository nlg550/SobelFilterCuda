#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_
#include "Image.h"

#include <cuda_runtime.h>

// Tile is a section/submatrix of the image
#define TILE_SIZE 32
#define MAX_VALUE(x, y) x > y ? x : y
#define MIN_VALUE(x, y) x < y ? x : y

// Built-in filter types
enum filter_type {
	SOBEL, BINOMIAL_X, BINOMIAL_Y
};

// Apply a given filter to the image
__global__ void apply_filter_cuda(const unsigned char *__restrict__ image_in, unsigned char *__restrict__ image_out, const int2 image_size, enum filter_type type);

#endif /* SRC_KERNEL_H_ */
