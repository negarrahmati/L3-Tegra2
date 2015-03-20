////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#ifndef _BILATERAL_FILTER_KERNEL_H_
#define _BILATERAL_FILTER_KERNEL_H_
#define APP_NAME "CUDA_CORE"

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG,  \
                                             APP_NAME, \
                                             __VA_ARGS__))
namespace gpu_bilateral_filter
{
    class BilateralFilter
    {
        public:
            BilateralFilter();
            ~BilateralFilter();
            // initialization of the textures here
            void initTexture(int width, int height,uint *hImage);
            // update coefficients for the gaussian based on radius
            void updateGaussian(float delta, int radius);
            // bilateral filter kernel launch
            void bilateralFilterRGBA(uint *hDest, int width, int height, float e_d, int radius, int iterations);
        private:
            // original iamge
            uint *d_image;
            // temp storage for intermediate results
            uint *d_temp;
            // final result of applying a box filter
            uint *d_result;
            size_t m_pitch;
    };
}//namespace gpu_bilateral_filter

#endif

