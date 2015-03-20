/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "helper_math.h"
#include "helper_functions.h"
#include "bilateral_kernel.h"
#include <android/log.h>

#ifndef _BILATERAL_FILTER_KERNEL_CU_
#define _BILATERAL_FILTER_KERNEL_CU_

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        LOGD("CUDA Runtime API error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__constant__ float cGaussian[64];   //gaussian array in device side
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.
    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

*/

namespace gpu_bilateral_filter
{

    //Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
    __device__ float euclideanLen(float4 a, float4 b, float d)
    {

        float mod = (b.x - a.x) * (b.x - a.x) +
                    (b.y - a.y) * (b.y - a.y) +
                    (b.z - a.z) * (b.z - a.z);

        return __expf(-mod / (2.f * d * d));
    }

    __device__ uint rgbaFloatToInt(float4 rgba)
    {
        rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
        rgba.y = __saturatef(fabs(rgba.y));
        rgba.z = __saturatef(fabs(rgba.z));
        rgba.w = __saturatef(fabs(rgba.w));
        return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
    }

    __device__ float4 rgbaIntToFloat(uint c)
    {
        float4 rgba;
        rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
        rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
        rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
        rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
        return rgba;
    }

    /*
        Bilateral filter kernel
        Parameters
        od - pointer to output data in global memory
        w  - image width
        h  - image height
        euclidean - euclidean delta
        r  - filter radius
    */
    __global__ void d_bilateral_filter(uint *od, int w, int h, float euclidean, int r)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if (x >= w || y >= h)
        {
            return;
        }

        float sum = 0.0f;
        float factor;
        float4 t = {0.f, 0.f, 0.f, 0.f};
        float4 center = tex2D(rgbaTex, x, y);

        for (int i = -r; i <= r; i++)
        {
            for (int j = -r; j <= r; j++)
            {
                float4 curPix = tex2D(rgbaTex, x + j, y + i);
                factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                         euclideanLen(curPix, center, euclidean);  // range factor
                t +=  factor * curPix;
                sum += factor;
            }
        }

        od[y * w + x] = rgbaFloatToInt(t/sum);
    }

    BilateralFilter::BilateralFilter()
    {
    }

    BilateralFilter::~BilateralFilter()
    {
        // free device memory
        checkCudaErrors(cudaFree(d_image));
        checkCudaErrors(cudaFree(d_result));
        checkCudaErrors(cudaFree(d_temp));
    }

    // Initialize the texture with input image data
    void BilateralFilter::initTexture(int width, int height, uint *hImage)
    {
        // Allocate memory on device
        checkCudaErrors(cudaMallocPitch(&d_image, &m_pitch, sizeof(uint)*width, height));
        checkCudaErrors(cudaMallocPitch(&d_temp, &m_pitch, sizeof(uint)*width, height));
        checkCudaErrors(cudaMallocPitch(&d_result,  &m_pitch, sizeof(uint)*width, height));
        // Copy image data from host memory to device global memory
        checkCudaErrors(cudaMemcpy2D(d_image, m_pitch, hImage, sizeof(uint)*width,
                                     sizeof(uint)*width, height, cudaMemcpyHostToDevice));
    }

    /*
        Because a 2D gaussian mask is symmetry in row and column,
        here only generate a 1D mask, and use the product by row
        and column index later.

        1D gaussian distribution :
            g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

        parameters:
        og - output gaussian array in global memory
        delta - the 2nd parameter 'd' in the above function
        radius - half of the filter size
                 (total filter size = 2 * radius + 1)
    */
    void BilateralFilter::updateGaussian(float delta, int radius)
    {
        float  fGaussian[64] = {0};

        for (int i = 0; i < 2*radius + 1; ++i)
        {
            float x = i-radius;
            fGaussian[i] = expf(-(x*x) / (2*delta*delta));
        }

        checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1),
                                           0, cudaMemcpyHostToDevice));
    }

    /*
        Perform 2D bilateral filter on image using CUDA

        Parameters:
        hDest - pointer to destination image in host memory
        width  - image width
        height - image height
        euclidean - euclidean delta
        radius - filter radius
        iterations - number of iterations
    */

    // RGBA version
    void BilateralFilter::bilateralFilterRGBA(uint *hDest,
                                              int width,
                                              int height,
                                              float euclidean,
                                              int radius,
                                              int iterations)
    {
        // Bind the array to the texture
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        checkCudaErrors(cudaBindTexture2D(0, rgbaTex, d_image, desc, width, height, m_pitch));

        for (int i=0; i<iterations; i++)
        {
            // Allocate grid size and block size
            dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
            dim3 blockSize(16, 16);

            // Apply simple bilateral filter
            d_bilateral_filter<<< gridSize, blockSize>>> (d_result, width, height, euclidean, radius);

            if (iterations > 1)
            {
                // Copy intermediate result to temp storage in global memory
                checkCudaErrors(cudaMemcpy2D(d_temp, m_pitch, d_result, sizeof(int)*width,
                                             sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
                // Bind temp result to texture
                checkCudaErrors(cudaBindTexture2D(0, rgbaTex, d_temp, desc, width, height, m_pitch));
            }
        }

        // copy back final result to host memory
        checkCudaErrors(cudaMemcpy2D(hDest, m_pitch, d_result, sizeof(uint)*width,
                                     sizeof(uint)*width, height, cudaMemcpyDeviceToHost));
    }
}
#endif // #ifndef _BILATERAL_FILTER_KERNEL_CU_

