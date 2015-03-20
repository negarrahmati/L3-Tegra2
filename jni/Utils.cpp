//----------------------------------------------------------------------------------
//
// Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------

#include <nv_file/nv_file.h>
#include "NativeApp.h"

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

typedef struct S_BITMAP_HEADER
{
    ushort fileType;
    uint fileSize;
    ushort reserved[2];
    uint bitmapPtr;
} __attribute__((packed)) BITMAP_HEADER;

typedef struct S_BMPV3_INFO_HEADER
{
    uint headerSize;
    uint bWidth;
    uint bHeight;
    ushort bitplanesNum;
    ushort bBpp;
    uint compression;
    uint bSize;
    uint hPixelPerMeter;
    uint vPixelPerMeter;
    uint colorsNum;
    uint imColorsNum;
} __attribute__((packed)) BMPV3_INFO_HEADER;

// loads 24-bit bmp files into 32-bit RGBA texture
unsigned int *LoadBMP(char const *name, int *width, int *height)
{
    NvFile *f = NvFOpen(name);

    if (f == 0)
    {
        LOGW("error opening %s!\n", name);
        return 0;
    }

    BITMAP_HEADER head;
    BMPV3_INFO_HEADER info;

    NvFRead(&head, sizeof(BITMAP_HEADER), 1, f);
    NvFRead(&info, sizeof(BMPV3_INFO_HEADER), 1, f);
    NvFSeek(f, head.bitmapPtr, SEEK_SET);

    *width = info.bWidth;
    *height = info.bHeight;
    uint *imageData = new uint[info.bWidth * info.bHeight];

    int scanSize = (info.bWidth * (info.bBpp >> 3) + 3) & ~0x3;
    uchar *rdata = new uchar[scanSize];

    for (int y = info.bHeight - 1; y >= 0; y--)
    {
        NvFRead(rdata, scanSize, 1, f);
        // shuffle rgb
        int index = y * info.bWidth;

        for (int i = 0; i < info.bWidth; i++)
        {
            imageData[index + i] = (rdata[i * 3] << 16)
                                   | (rdata[i * 3 + 1] << 8) | rdata[i * 3 + 2] | 0xff000000;
        }
    }

    NvFClose(f);
    delete[] rdata;
    LOGI("loaded %ix%i input image", info.bWidth, info.bHeight);

    return imageData;
}
