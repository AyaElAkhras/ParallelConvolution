#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#define cimg_display 0
#include "CImg.h"
using namespace cimg_library;

#define TILE_WIDTH 32
#define MASK_WIDTH 3

// Masks
float blur_mask[MASK_WIDTH][MASK_WIDTH] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
float emboss_mask[MASK_WIDTH][MASK_WIDTH] = {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};
float outline_mask[MASK_WIDTH][MASK_WIDTH] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
float sharpen_mask[MASK_WIDTH][MASK_WIDTH] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
float left_sobel_mask[MASK_WIDTH][MASK_WIDTH] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
float right_sobel_mask[MASK_WIDTH][MASK_WIDTH] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
float top_sobel_mask[MASK_WIDTH][MASK_WIDTH] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
float bottom_sobel_mask[MASK_WIDTH][MASK_WIDTH] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};


// Enumerating operation keys
enum operation_name{
	BLUR, 
	EMBOSS, 
	OUTLINE,
	SHARPEN,
	LEFT_SOBEL,
	RIGHT_SOBEL,
	TOP_SOBEL,
	BOTTOM_SOBEL
};


// Constant Memory Variable to keep the user's specified mask type
__constant__ float const_mask[MASK_WIDTH][MASK_WIDTH];


__global__ void ConvKernel(float* input, float* output, int img_width, int img_height){
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(blockIdx.x==0&&blockIdx.y==1)
		col = col+1-1;
	__shared__ float input_shared[TILE_WIDTH][TILE_WIDTH];
	if(row < img_height && col < img_width){
		input_shared[threadIdx.y][threadIdx.x] = input[row*img_width + col];
	}
	else{
		input_shared[threadIdx.y][threadIdx.x] = 0.0;    // Extra threads will load 0 to the shared memory
	}

	// wait until all threads of the block are done filling the shared memory
	__syncthreads();

	// internal cells boundary indices for the tile
	int this_tile_start_row = blockIdx.y * blockDim.y;
	int this_tile_start_col = blockIdx.x * blockDim.x;
	int next_tile_start_row = (blockIdx.y + 1) * blockDim.y<img_height?(blockIdx.y + 1) * blockDim.y:img_height;
	int next_tile_start_col = (blockIdx.x + 1) * blockDim.x<img_width?(blockIdx.x + 1) * blockDim.x:img_width;

	int i_start = row - (MASK_WIDTH / 2);
	int j_start = col - (MASK_WIDTH / 2);
	float out_val = 0.0;
	int actual_j ,actual_i;
	for(int i = 0; i < MASK_WIDTH; i++)
	{
		for(int j = 0; j < MASK_WIDTH; j++)
		{
			actual_i = i_start + i;
			actual_j = j_start + j;
			// Deciding whether the current indices correspond to internal or halo cells
			if(actual_j >= this_tile_start_col && actual_j < next_tile_start_col
					&& actual_i >= this_tile_start_row && actual_i < next_tile_start_row){  // internal cell
				out_val += input_shared[threadIdx.y + i - (MASK_WIDTH / 2)][threadIdx.x + j - (MASK_WIDTH / 2)] * const_mask[i][j];
			}

			else{  // halo cell, load from global memory
				// Checking Image Boundary Conditions
				// Checks on row
				// if none of the above conditions is satisfied, actual_j is valid and no need to change it
				if(actual_i < 0)
					actual_i = 0;

				else
				{
					if(actual_i >= img_height)
						actual_i = img_height - 1;
				}
					// if none of the above conditions is satisfied, actual_i is valid and no need to change it
					// Checks on col
				if(actual_j < 0)
					actual_j = 0;

				else
				{
					if(actual_j >= img_width)
						actual_j = img_width - 1;
				}
				// End of Boundary Conditions
				out_val += input[actual_i * img_width + actual_j] * const_mask[i][j];
			}

		}
	}


	if(row < img_height && col < img_width){
		 output[row*img_width + col] = out_val;
	}

}


// Wrapper for the Kernel
double parallelConv(float* input, float* out, float chosen_mask[][MASK_WIDTH], int img_width, int img_height){
	int img_bytes_size = img_width * img_height * sizeof(float);
	float *d_input;
	float *d_output;
	cudaError_t err;
	clock_t start, stop;
	double time_spent; 

	err = cudaMemcpyToSymbol(const_mask, chosen_mask, MASK_WIDTH*MASK_WIDTH*sizeof(float));
	if (err!= cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**) &d_input, img_bytes_size);
	if (err!= cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_input, input, img_bytes_size, cudaMemcpyHostToDevice);
	if (err!= cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void**) &d_output, img_bytes_size);
	if (err!= cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	dim3 gridDim(ceil(img_width/float(TILE_WIDTH)), ceil(img_height/float(TILE_WIDTH)), 1);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	// Kernel Invocation
	start = clock();
	ConvKernel<<<gridDim, blockDim>>>(d_input, d_output, img_width, img_height);
	err = cudaDeviceSynchronize();
	if (err!= cudaSuccess) { 
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); 
		exit(EXIT_FAILURE);  
	}
	stop = clock();
	time_spent = (double)(stop - start) / CLOCKS_PER_SEC; 

	err = cudaMemcpy(out, d_output, img_bytes_size, cudaMemcpyDeviceToHost);
	if (err!= cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_input);
	cudaFree(d_output);

	return time_spent; 
}


void sequentialConv(float* input, float* output, float chosen_mask[][MASK_WIDTH], int img_width, int img_height){
	// loop on every single pixel in the img
	for(int row = 0; row < img_height; row++){
		int start_i = row - (MASK_WIDTH / 2);
		for(int col = 0; col < img_width; col++){
			int start_j = col - (MASK_WIDTH / 2);

			float out_val = 0.0;
			int proper_i, proper_j;

			// Scan Mask elements over the input element and its surroundings
			for(int i = 0; i < MASK_WIDTH; i++){
				for(int j = 0; j < MASK_WIDTH; j++){
					// Checking on any Boundary Conditions
						// Checking on Row
					if((start_i+i) < 0)
						proper_i = 0;
					else{
						if((start_i+i) >= img_height)
							proper_i = img_height - 1;

						else
							proper_i = start_i+i;
					}
						// Checking on Col
					if((start_j+j) < 0)
						proper_j = 0;
					else{
						if((start_j+j) >= img_width)
							proper_j = img_width - 1;

						else
							proper_j = start_j+j;
					}
					// End of Boundary Conditions Checks

					out_val += input[proper_i * img_width + proper_j] * chosen_mask[i][j];
				}
			}
			output[row*img_width+col] = out_val;
		}
	}
}

void compareOutputs(float* seq_out, float* parallel_out, int img_width, int img_height){  // For checking the parallel against sequential
	int check_flag = 1; 
	for(int i=0; i<img_height*img_width ;i++)
		if(seq_out[i] != parallel_out[i]){
			check_flag = 0;
			break;
		}

	if(check_flag == 1){
		printf("The 2 implementations produced the same exact output\n");
	}
	else{
		printf("The 2 implementations produced different outputs\n");
	}
}

void printChosenMask(float chosen_mask[][MASK_WIDTH]){   // For Debugging 
	for(int i = 0; i < MASK_WIDTH; i++){
		for(int j = 0; j < MASK_WIDTH; j++)
				printf("  %f", chosen_mask[i][j]);

		printf("\n");

	}
}

void setMask(float from[][MASK_WIDTH], float to[][MASK_WIDTH]){
		for(int i = 0; i < MASK_WIDTH; i++){
			for(int j = 0; j < MASK_WIDTH; j++)
				to[i][j] = from[i][j];
		}
}

int choose_mask(operation_name operation_key, float chosen_mask[][MASK_WIDTH]){
	int err_code = 0; 
	switch(operation_key){
		case BLUR:
			setMask(blur_mask, chosen_mask);
			break;

		case EMBOSS:
			setMask(emboss_mask, chosen_mask);
			break;

		case OUTLINE:
			setMask(outline_mask, chosen_mask);
			break;

		case SHARPEN:
			setMask(sharpen_mask, chosen_mask);
			break;

		case LEFT_SOBEL:
			setMask(left_sobel_mask, chosen_mask);
			break;

		case RIGHT_SOBEL:
			setMask(right_sobel_mask, chosen_mask);
			break;

		case TOP_SOBEL:
			setMask(top_sobel_mask, chosen_mask);
			break;

		case BOTTOM_SOBEL:
			setMask(bottom_sobel_mask, chosen_mask);
			break;

		default:
			err_code = 1;    // denoting that the chosen operation_code isn't supported 
	}

	return err_code;	
}

unsigned long get_operations_num(int img_width ,int img_height){
	return (2 * MASK_WIDTH * MASK_WIDTH * img_width * img_height);
}


int main()
{
		// Declarations
	char input_img_name[30], output_img_name[30];  // Assume that the images names won't exceed 30 characters 
	operation_name operation_key;    // enum type for convenience 
	float chosen_mask[MASK_WIDTH][MASK_WIDTH];
	int err_code;   // returned by the choose_mask function and is raised to 1 if the inputted operation_code isn't supported
	float* parallel_out, *sequential_out;

	clock_t start, stop;
	double kernel_time;
	double gflops_seq = -1;
	double gflops_parallel_wrapper = -1;
	double gflops_parallel_kernel = -1;


	const int gigabyte = 1073741824;  // to convert from byte to GB divide by this constant



		// Handling input from user 
	printf("Please enter the input image name\n");
	scanf("%s", input_img_name);

	printf("Please enter the output image name\n");
	scanf("%s", output_img_name);

	do{
		printf("Please enter the operation code as a digit from 0 to 7\n");
		scanf("%d", &operation_key);
		err_code = choose_mask(operation_key, chosen_mask);
	}while(err_code == 1);   // keep doing so as long as the operation code isn't supported 
	
		// The following lines are for debugging to confirm that the input was picked successfully 
	printf("\n");
	printf("The input image name is: %s\n", input_img_name);
	printf("The output image name is: %s\n", output_img_name);
	printf("The chosen mask is: \n");
	printChosenMask(chosen_mask);
	printf("\n");
		// Done handling input from user 
	
		// Start reading the input image 
	CImg<float> image(input_img_name);
	size_t img_width = image.width();
	size_t img_height = image.height();
	size_t channels = image.spectrum();
	size_t dimensions = image.depth();
	printf("Width: %lu, Height: %lu, Channels: %lu, Dim: %lu\n", img_width, img_height, channels, dimensions);
	float* img_data = image.data();

		// Memory Allocations
	parallel_out = (float*)malloc(img_width * img_height * sizeof(float));
	sequential_out = (float*)malloc(img_width * img_height * sizeof(float));
	

		// Running the implementations 
	start = clock();
	kernel_time = parallelConv(img_data, parallel_out, chosen_mask, img_width, img_height);
	stop = clock(); 
	double time_spent_parallel = (double)(stop - start) / CLOCKS_PER_SEC; 

	start = clock();
	sequentialConv(img_data, sequential_out, chosen_mask, img_width, img_height);
	stop = clock(); 
	double time_spent_sequential = (double)(stop - start) / CLOCKS_PER_SEC; 

	printf("\n");
	compareOutputs(sequential_out, parallel_out, img_width, img_height);


		// Preparing output files names 
	char par_output_img_name[35] = "par_";
	char seq_output_img_name[35] = "seq_";

		// Saving the images resulting from each of the 2 implementations 
	CImg<float> output_parallel_image(parallel_out, img_width, img_height, dimensions, channels);
	output_parallel_image.save(strcat(par_output_img_name, output_img_name));

	CImg<float> output_sequential_image(sequential_out, img_width, img_height, dimensions, channels);
	output_sequential_image.save(strcat(seq_output_img_name, output_img_name));

	printf("\n");
	printf("The Tile Width is %d\n", TILE_WIDTH);
	printf("\n");
	printf("The execution time for the sequential implementation is %f seconds\n", time_spent_sequential);
	printf("The execution time for the parallel implementation is %f seconds\n", time_spent_parallel);
	printf("The execution time for the kernel alone is %f seconds\n", kernel_time);

	// Retrieve and print the total number of operations
	unsigned long total_number_of_operations = get_operations_num(img_width, img_height);

	// Calculating the GFLOPs
	if(time_spent_sequential != 0)
		gflops_seq = (total_number_of_operations / time_spent_sequential) / gigabyte;

	if(time_spent_parallel != 0)
		gflops_parallel_wrapper = (total_number_of_operations / time_spent_parallel) / gigabyte;

	if(kernel_time != 0)
		gflops_parallel_kernel = (total_number_of_operations / kernel_time) / gigabyte;


	printf("\n");
	printf("The total number of operations is %d\n", total_number_of_operations);
	printf("\n");
	printf("The value of GFLOPS in sequential implementation is  %f GLOPS\n", gflops_seq);
	printf("The value of GFLOPS in parallel wrapper implementation is  %f GLOPS\n", gflops_parallel_wrapper);
	printf("The value of GFLOPS in parallel kernel implementation is  %f GLOPS\n", gflops_parallel_kernel);


		// De-allocate Memory 
	free(parallel_out);
	free(sequential_out);

	return 0;
}

