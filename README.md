# ParallelConvolution
This program is done as a part of GPU Computing course. 

# Description
Parallel implementation for the 2D convolution operation using CUDA C, thus requiresa CUDA enabled device. It works on Grey scale images of arbitrary size. It supports 8 types of filters: blur, emboss, outline, sharpen, and sobel (left, right, top, and bottom). It utilizes Shared Memory for stroing the input image matrix for Coalesced Memory accesses, and Constant Memory for storing filters. 

# Dependencies
CImg C library is used to open and save images.

# Usage
The program expects the following inputs from users:
1. Input Image name
2. Output Imgae name
3. Type of operation, the following convention is adopted
	0 for Blur, 1 for Emboss, 2 for Outline, 3 for Sharpen, 4 for Left Sobel, 5 for Right Sobel, 6 for Top Sobel, 7 for Bottom Sobel 
