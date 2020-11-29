#include "kernel.h"

/********************************************************************
 
 Sobel (x) : [-1    0    1]
 	 	 	 [-2    0    2]
 	 	 	 [-1    0    1]
 
 Sobel (y) : [ 1	2 	 1]
 	 	 	 [ 0 	0 	 0]
 	 	 	 [-1   -2 	-1]
 
 Binomial (x) : [0.25 0.5 0.25]
 
 Binomial (y) : [0.25] 
 	 	 	 	[0.5 ]
 	 	 	 	[0.25]
 
 ********************************************************************/

// Only works with kernel that have size less or equal than 3x3
__global__ void apply_filter_cuda(const unsigned char *__restrict__ image_in, unsigned char *__restrict__ image_out, const int2 image_size, enum filter_type type)
{
	unsigned char out;

	// Global index in which the tile begins
	int2 begin_idx;
	begin_idx.x = TILE_SIZE * blockIdx.x;
	begin_idx.y = TILE_SIZE * blockIdx.y;
	
	// Allocate a tile surrounded with ghost cells
	__shared__ unsigned char tile_in[TILE_SIZE + 2][TILE_SIZE + 2]; 
	
	// Load the global memory (Image) into the local memory (Tile)
	for(int j = threadIdx.y; j < (TILE_SIZE + 2); j += blockDim.y)
	{
		for(int i = threadIdx.x; i < (TILE_SIZE + 2); i += blockDim.x)
		{	
			// In case of out-of-bounds indexes, copy the nearest valid value
			int idx_x = MAX_VALUE(MIN(begin_idx.x + i - 1, image_size.x - 1), 0);
			int idx_y = MAX_VALUE(MIN(begin_idx.y + j - 1, image_size.y - 1), 0);
			tile_in[j][i] = image_in[idx_x + idx_y * image_size.x];
		}
	}
	
	__syncthreads(); // Local Barrier
	
	// Apply the filter to the image using the values store in the tile 
	for(int y = threadIdx.y; y < TILE_SIZE; y += blockDim.y)
	{
		for(int x = threadIdx.x; x < TILE_SIZE; x += blockDim.x)
		{			
			int idx_x = begin_idx.x + x;
			int idx_y = begin_idx.y + y;
			
			switch (type)
			{
				case SOBEL:
					short sum;
					sum  = tile_in[y][x + 2] + 2 * tile_in[y + 1][x + 2] + tile_in[y + 2][x + 2];
					sum -= tile_in[y][x] 	+ 2 * tile_in[y + 1][x] 	+ tile_in[y + 2][x];
					sum += tile_in[y][x] 	+ 2 * tile_in[y][x + 1]		+ tile_in[y][x + 2];
					sum -= tile_in[y + 2][x] + 2 * tile_in[y + 2][x + 1] + tile_in[y + 2][x + 2];
					
					if(sum < 0) out = 0;
					else if(sum >= 0xFF) out = 0xFF;
					else out = (unsigned char) sum;

					break;
					
				case BINOMIAL_X:
					out = 0.25f * tile_in[y + 1][x] + 0.5f * tile_in[y + 1][x + 1] + 0.25f * tile_in[y + 1][x + 2];
					break;
					
				case BINOMIAL_Y:
					out = 0.25f * tile_in[y][x + 1] + 0.5f * tile_in[y + 1][x + 1] + 0.25f * tile_in[y + 2][x + 1];
					break;
			}
			
			// Only write values that are inside the image
			if(idx_x >= 0 && idx_x < image_size.x && idx_y >= 0 && idx_y < image_size.y)
				image_out[idx_x + idx_y * image_size.x] = out;
		}
	}
}

__host__ void Image::apply_sobel_filter()
{
	// Calculate the dimension of the Grid/Block
	dim3 num_blocks(std::ceil((float) size.x / TILE_SIZE), std::ceil((float) size.y / TILE_SIZE));
	dim3 num_threads(32, 32);

	// Prefetch the data to the device to avoid page faults during the kernel execution
	cudaMemPrefetchAsync(dev_ptr, size.x * size.y * sizeof(unsigned char), 0, NULL);
	
	apply_filter_cuda<<<num_blocks, num_threads>>>(dev_ptr, dev_ptr, size, SOBEL);
	
	// Synchronize the execution in the host and the device
	cudaDeviceSynchronize(); 
}

__host__ void Image::apply_binomial_filter(const int n_pass)
{
	// Calculate the dimension of the Grid/Block
	dim3 num_blocks(std::ceil((float) size.x / TILE_SIZE), std::ceil((float) size.y / TILE_SIZE));
	dim3 num_threads(32, 32);

	// Prefetch the data to the device to avoid page faults during the kernel execution
	cudaMemPrefetchAsync(dev_ptr, size.x * size.y * sizeof(unsigned char), 0, NULL);
	
	for(int n = 0; n < n_pass; n++)
	{
		apply_filter_cuda<<<num_blocks, num_threads>>>(dev_ptr, dev_ptr, size, BINOMIAL_X);
		apply_filter_cuda<<<num_blocks, num_threads>>>(dev_ptr, dev_ptr, size, BINOMIAL_Y);
	}
	
	// Synchronize the execution in the host and the device
	cudaDeviceSynchronize();
}