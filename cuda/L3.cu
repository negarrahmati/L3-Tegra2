/*
 * L3.cu
 *
 *  Created on: Mar 12, 2015
 *      Author: mvc
 */

#include "L3.h"

L3::L3() {
	// TODO Auto-generated constructor stub

}

L3::~L3() {
	// TODO Auto-generated destructor stub
}

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>
#include "cuda.h"
#include <cuda_runtime.h>

using namespace std;

// Define constant parameters
#define cfa_size      4
#define num_filter    5
#define patch_size    9
#define border_size   4
#define image_width   720
#define image_height  1280
#define voltage_max   0.9734
#define lum_list_size 20
#define num_out       3
#define low           0.95
#define high          1.15
#define sat_levels    19

#define num_frames    100

/* Cuda function - L3Render
 
 Compute mean for each channel
 
 Inputs:
 out_image       - pre-allocated space for output (xyz) image
 image           - original image
 cfa             - cfa pattern, should be 0-indexed
 lum_list        - luminance list
 sat_list        - saturation list
 flat_filters    - filters for flat regions
 texture_filters - filters for texture regions
 */
__global__ 
void L3Render(float* const out_image,
              float  const * const image,
              float  const * const lum_list,
              float  const * const sat_list,
              float  const * const flat_filters,
              float  const * const texture_filters,
              float  const * const flat_threshold_list)
{
    // Find pixel position
    const int row = blockIdx.x;
    const int col = threadIdx.x;
	const size_t out_pixel_index = num_out*(row + col * image_height);
    
    // Check pixel range
    if ((row < border_size) ||
        (row >= image_height - border_size) ||
        (col < border_size) ||
        (col >= image_width - border_size)){
			return;
	}

    // Compute patch type
    const unsigned short patch_type[] = {row % cfa_size, col % cfa_size};          // patch type
	const unsigned short cfa[] = {1, 0, 1, 3, 4, 1, 2, 1, 1, 3, 1, 0, 2, 1, 4, 1}; // cfa pattern

	// Store patch data in image to local memory
	float patch_data[patch_size*patch_size];

	
    
    // Compute mean for each channel
    float channel_mean[num_filter] = {0.0};
    unsigned short channel_count[num_filter] = {0};
    unsigned short cfa_index[patch_size * patch_size];
    
	unsigned short index = 0;
	unsigned short col_index = (col - border_size) % cfa_size;
	unsigned short row_index = (row - border_size) % cfa_size;
	size_t pixel_index = (row - border_size) + (col - border_size)*image_height;
	bool is_sat[num_filter] = {false};

    for (short jj = -border_size; jj <= border_size; jj++){
		unsigned short j_index = col_index * cfa_size;
		unsigned short i_index = row_index;
        for (short ii = -border_size; ii <= border_size; ii++){
            cfa_index[index] = i_index + j_index;
            channel_count[cfa[cfa_index[index]]] += 1;
			patch_data[index] = image[pixel_index];
			channel_mean[cfa[cfa_index[index]]] += patch_data[index];
			is_sat[cfa[cfa_index[index]]] |= (patch_data[index] > voltage_max);
			index++; pixel_index++; i_index = (i_index + 1) % cfa_size;
        }
		pixel_index += image_height-2*border_size-1;
		col_index = (col_index + 1) % cfa_size;
    }

    
    // Compute channel mean luminance
	// Channel mean luminance is defined as the mean of channel_mean
    float lum_mean = 0;
    for (int ii = 0; ii < num_filter; ii++) {
        channel_mean[ii] /= channel_count[ii];
        lum_mean += channel_mean[ii];
    }
    lum_mean /= num_filter;
    
    // Convert luminance to luminance index
    // Binary search might be faster for large nubmer of luminance levels
	// But this difference can be ignored when we have only around 20 luminance levels
    unsigned short lum_index = lum_list_size - 1;
    for (int ii = 0; ii < lum_list_size; ii++) {
        if (lum_mean < lum_list[ii]) {
            lum_index = ii;
            break;
        }
    }
    
    // Compute saturation type
    unsigned short sat_type = 0; // sat_type is the encoded saturation type
    unsigned short sat_index;    // sat_index is the number found with sat_list
    // const unsigned short sat_list_size = (1 << num_filter);
    for (int ii = num_filter - 1; ii >= 0; ii --)
        sat_type = (sat_type << 1) + is_sat[ii]; // (channel_mean[ii] > voltage_max);
    
	const float *cur_sat_list = sat_list + ((patch_type[1] * cfa_size + patch_type[0]) << num_filter);
    sat_index = cur_sat_list[sat_type];
    
    // Find nearest sat_type for missing ones
	const unsigned short sat_list_size = (1 << num_filter);
    if (sat_index == 0){
        float min_cost = 10000; // Init min cost to some arbitrarily large value
        for (int ii = 0; ii < sat_list_size; ii++) {
			if (cur_sat_list[ii] != 0){
				// compute cost
				float cur_cost = 0;
				unsigned short sat_diff = (ii ^ sat_type);
				for (int jj = 0; sat_diff > 0; jj++) {
					if ((sat_diff & 1) > 0)
						cur_cost += fabsf(channel_mean[jj] - voltage_max);
					sat_diff = (sat_diff >> 1);
				}
				if (cur_cost < min_cost) {
					min_cost = cur_cost;
					sat_index = cur_sat_list[ii];
				}
			}
		}
    }
    sat_index--; // make sat_index 0-indexed
    
    // Compute image contrast
    // Assume image_contrast array has been allocated as zeros
    float image_contrast = 0;
	for (index = 0; index < patch_size * patch_size; index++)
		image_contrast += fabsf(patch_data[index] - channel_mean[cfa[cfa_index[index]]]);
	image_contrast /= (patch_size * patch_size);
    
    // Determine flat or texture
    const int threshold_index  = ((sat_index * lum_list_size + lum_index) * cfa_size + patch_type[1]) * cfa_size + patch_type[0];
    const float flat_threshold = flat_threshold_list[threshold_index];

	// Apply filter to patch
	const float *filter;
	float out_data[num_out] = {0};
	unsigned short filter_index;
	unsigned int filter_offset = threshold_index * num_out * patch_size * patch_size;
    if (image_contrast < flat_threshold * low) { // flat region
        filter = flat_filters + filter_offset;	
		for (index = 0, filter_index = 0; index < patch_size * patch_size; index++, filter_index += 3){
			out_data[0] += patch_data[index] * filter[filter_index];
			out_data[1] += patch_data[index] * filter[filter_index+1];
            out_data[2] += patch_data[index] * filter[filter_index+2];
		}
    }
    else if (image_contrast > flat_threshold * high) { // texture region
        filter = texture_filters + filter_offset;
		for (index = 0, filter_index = 0; index < patch_size * patch_size; index++, filter_index += 3){
			out_data[0] += patch_data[index] * filter[filter_index];
			out_data[1] += patch_data[index] * filter[filter_index+1];
            out_data[2] += patch_data[index] * filter[filter_index+2];
		}
    }
    else { // transition region
        const float weights = (image_contrast / flat_threshold - low) / (high - low);
        filter = flat_filters + filter_offset;
        const float* filter_texture = texture_filters + filter_offset;
		for (index = 0, filter_index = 0; index < patch_size * patch_size; index ++, filter_index += 3){
			out_data[0] += patch_data[index] * (filter[filter_index]   * weights + filter_texture[filter_index] * (1 - weights));
            out_data[1] += patch_data[index] * (filter[filter_index+1] * weights + filter_texture[filter_index+1] * (1 - weights));
			out_data[2] += patch_data[index] * (filter[filter_index+2] * weights + filter_texture[filter_index+2] * (1 - weights));
		}
    }
	out_image[out_pixel_index] = out_data[0];
	out_image[out_pixel_index + 1] = out_data[1];
	out_image[out_pixel_index + 2] = out_data[2];
}

// Main routine
int L3::L3_main(void)
{
	LOGD("hello");
    // Init parameters
    float * out_image, * out_image_d; // pointer to rendered image
    float * image, * image_d; // pointer to input raw image data
    float  * lum_list, * lum_list_d; // pointer to luminance list
    float  * sat_list, * sat_list_d; // pointer to saturation list
    float  * flat_filters, * flat_filters_d; // pointer to filters in flat region
    float  * texture_filters, * texture_filters_d; // pointer to filters in texture region
    float  * flat_threshold_list, * flat_threshold_list_d; // the list of thresholds of determining a patch is flat or not
    
    const unsigned short sat_list_size = (1 << num_filter)*cfa_size*cfa_size;
	const unsigned int flat_filters_size = num_out * patch_size * patch_size * lum_list_size * cfa_size * cfa_size * sat_levels;
    const unsigned int texture_filters_size = flat_filters_size;
	const unsigned int flat_threshold_list_size = lum_list_size*cfa_size*cfa_size*sat_levels;
    
	LOGD("hello 1");
    // Allocate spaces in main memory
    image = (float*)malloc(sizeof(float)*image_width*image_height);
    out_image = (float*) malloc(sizeof(float)*image_height*image_width*num_out);
    lum_list = (float*) malloc(sizeof(float)*lum_list_size);
    sat_list = (float*) malloc(sizeof(float)*sat_list_size);
    flat_filters = (float*) malloc(sizeof(float) * flat_filters_size);
    texture_filters = (float*)malloc(sizeof(float)* texture_filters_size);
	flat_threshold_list = (float*)malloc(sizeof(float)*flat_threshold_list_size);
    
	LOGD("hello 2");
    // Load data from files
    FILE* pf;
   
    pf = fopen("/sdcard/cudadata/lum_list.dat", "rb"); // luminance list
   
    	
	LOGD("hello 3");
    fread(lum_list, sizeof(float), lum_list_size, pf);
    
	LOGD("hello 4");
	fclose(pf);
    
    pf = fopen("/sdcard/cudadata/sat_list.dat", "rb"); // saturation list
    fread(sat_list, sizeof(float), sat_list_size, pf);
    fclose(pf);
    
    pf = fopen("/sdcard/cudadata/flat_filters.dat", "rb"); //flat filters
    fread(flat_filters, sizeof(float), flat_filters_size, pf);
    fclose(pf);
    
    pf = fopen("/sdcard/cudadata/texture_filters.dat", "rb"); // texture filters
    fread(texture_filters, sizeof(float), texture_filters_size, pf);
    fclose(pf);
    
    pf = fopen("/sdcard/cudadata/flat_threshold_list.dat", "rb"); // flat threshold list
	fread(flat_threshold_list, sizeof(float), flat_threshold_list_size, pf);
    fclose(pf);
    
    // Allocate spaces in GPU
    cudaMalloc((void **) & out_image_d, sizeof(float)*image_width*image_height*num_out);
	cudaMalloc((void **) & image_d, sizeof(float)*image_height * image_width);
    cudaMalloc((void **) & lum_list_d, sizeof(float)*lum_list_size);
    cudaMalloc((void **) & sat_list_d, sizeof(float)*sat_list_size);
    cudaMalloc((void **) & flat_filters_d, sizeof(float)*flat_filters_size);
    cudaMalloc((void **) & texture_filters_d, sizeof(float)*texture_filters_size);
	cudaMalloc((void **) & flat_threshold_list_d, sizeof(float)*flat_threshold_list_size);
    
    // Copy data to GPU
    cudaMemcpy(lum_list_d, lum_list, sizeof(float)*lum_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(sat_list_d, sat_list, sizeof(float)*sat_list_size, cudaMemcpyHostToDevice);
    cudaMemcpy(flat_filters_d, flat_filters, sizeof(float)*flat_filters_size, cudaMemcpyHostToDevice);
    cudaMemcpy(texture_filters_d, texture_filters, sizeof(float)*texture_filters_size, cudaMemcpyHostToDevice);
	cudaMemcpy(flat_threshold_list_d, flat_threshold_list, sizeof(float)*flat_threshold_list_size, cudaMemcpyHostToDevice);

	char *fName = new char[100];
	struct timeval tm1, tm2;
	
	for (int fIndex = 0; fIndex < num_frames; fIndex ++ ){
		//Runtime including IO
		//gettimeofday(&tm1, NULL);
		//LOGD("frame # %d", fIndex);
		// show debug info
		printf("Processing frame %d...\n", fIndex);

		// Load image
		// sprintf(fName, "./video/output_%07d.dat", fIndex);
		sprintf(fName, "/sdcard/cudadata/raw_image.dat", fIndex);
		pf = fopen(fName, "rb"); // image raw data
		fread(image, sizeof(float), image_width * image_height, pf);
		fclose(pf);
		
		cudaMemcpy(image_d, image, sizeof(float)*image_height*image_width, cudaMemcpyHostToDevice);
		cudaMemset(out_image_d, 0, image_width*image_height*num_out*sizeof(float));
    
    	gettimeofday(&tm1, NULL);
		
		// Do computation in GPU
		L3Render<<<image_height, image_width>>>(out_image_d, image_d, lum_list_d, sat_list_d, flat_filters_d, texture_filters_d, flat_threshold_list_d);
    	
		// Copy back to main memory
		cudaMemcpy(out_image, out_image_d, sizeof(float)*image_width*image_height*num_out, cudaMemcpyDeviceToHost);
    	gettimeofday(&tm2, NULL);
		unsigned long long t = 1000000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec);
		LOGD("time lapse: %llu", t);
		
    	
		// Write rendered image to file
		// sprintf(fName, "./video_out/frame%07d.dat", fIndex);
		sprintf(fName, "/sdcard/cudadata/raw_image.dat", fIndex);
		pf = fopen(fName, "wb");
		fwrite(out_image, sizeof(float), image_height * image_width * num_out, pf);
		fclose(pf);
		
		//runtime including IO
		/*gettimeofday(&tm2, NULL);
		unsigned long long t = 1000 * (tm2.tv_sec - tm1.tv_sec) + (tm2.tv_usec - tm1.tv_usec) / 1000;
		LOGD("time lapse: %llu", t);*/
	}
	LOGD("done");
	
    // Cleanup and return
    free(out_image); cudaFree(out_image_d);
    free(image); cudaFree(image_d);
    free(lum_list); cudaFree(lum_list_d);
    free(sat_list); cudaFree(sat_list_d);
    free(flat_filters); cudaFree(flat_filters_d);
    free(texture_filters); cudaFree(texture_filters_d);
    free(flat_threshold_list); cudaFree(flat_threshold_list_d);
    
    LOGD("done 4");
	//cudaDeviceReset();
	LOGD("blah 4");
	
    return 0;
}

