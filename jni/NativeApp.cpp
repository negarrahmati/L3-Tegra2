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

#include <nv_shader/nv_shader.h>
#include <nv_file/nv_file.h>
#include "NativeApp.h"
#include "Utils.h"
#include "bilateral_kernel.h"

//Negar
#include "L3.h"

#define REFRESH_DELAY     10 //ms
#define MIN_EUCLIDEAN_D   0.01f
#define MAX_EUCLIDEAN_D   5.0f
#define MAX_FILTER_RADIUS 25

NativeApp::NativeApp(android_app *app, NvEGLUtil *egl) : mEgl(egl),
    mNativeAppInstance(app),
    mIterations(1),
    mGaussianDelta(4),
    mEuclideanDelta(0.1f),
    mFilterRadius(1),
    mFilterSign(1),
    mProcessedData(NULL)
{
    app->userData = this;
    app->onAppCmd = HandleCommand;
    app->onInputEvent = HandleInput;

    mCurrentApplicationState = INITIALIZATION;
    mScreenPressed = false;
}

NativeApp::~NativeApp()
{
    delete [] mProcessedData;
}

/**
 *  Called every frame
 */
void NativeApp::renderFrame(void)
{
    if (!mEgl->isReadyToRender(true))
    {
        return;
    }

    if (mEgl->checkWindowResized())
    {
        glViewport(0, 0, mEgl->getWidth(), mEgl->getHeight());
    }

    switch (mCurrentApplicationState)
    {
        case INITIALIZATION:
            {
                // Initialize shader loader
                nv_shader_init(mNativeAppInstance->activity->assetManager);

                // Initialize file loader
                NvFInit(mNativeAppInstance->activity->assetManager);

                // Load plain texture render program
                mPlainTextureProgram = nv_load_program("plain");

                // Load input texture
                glGenTextures(1, &mInputImageTexture);
                glBindTexture(GL_TEXTURE_2D, mInputImageTexture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                uint *textureData = LoadBMP("nature_monte.bmp", &mTextureWidth, &mTextureHeight);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mTextureWidth, mTextureHeight, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, textureData);

                // update the GausianDelta based on filter radius
                bilateralFilter.updateGaussian(mGaussianDelta, mFilterRadius);

                // initialize bilateral filter with input texture data
                bilateralFilter.initTexture(mTextureWidth, mTextureHeight, textureData);

                // generate output texture
                glGenTextures(1, &mOutputImageTexture);
                glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mTextureWidth, mTextureHeight, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, 0);

                // output from bilateral filter is stored in mProcessedData
                mProcessedData = new unsigned int[mTextureWidth * mTextureHeight];

                mCurrentApplicationState = MAIN_LOOP;
                break;
            }

        case MAIN_LOOP:
            {
                // clear color buffer
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);


                //Negar. invoke L3 main function
                L3* l3 = new L3();
                l3->L3_main();

                // invoke bilateral filter as CUDA kernel
                bilateralFilter.bilateralFilterRGBA(mProcessedData,
                                                    mTextureWidth,
                                                    mTextureHeight,
                                                    mEuclideanDelta,
                                                    mFilterRadius,
                                                    mIterations);

                // vary filter radius
                mFilterRadius += mFilterSign;

                if (mFilterRadius > MAX_FILTER_RADIUS)
                {
                    mFilterSign = -1;
                    mFilterRadius = MAX_FILTER_RADIUS;
                }
                else if (mFilterRadius < 1)
                {
                    mFilterSign = 1;
                    mFilterRadius = 0;
                }

                // update GaussianDelta according to filter radius
                bilateralFilter.updateGaussian(mGaussianDelta, mFilterRadius);

                static float factor = 1.02f;

                // vary Euclidean Delta
                if (mEuclideanDelta > MAX_EUCLIDEAN_D)
                {
                    factor = 1/1.02f;
                }

                if (mEuclideanDelta < MIN_EUCLIDEAN_D)
                {
                    factor = 1.02f;
                }

                mEuclideanDelta *= factor;

                // bind output texture
                glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mTextureWidth, mTextureHeight, GL_RGBA,
                                GL_UNSIGNED_BYTE, (unsigned char *)mProcessedData);

                // Per vertex x and y coordinates inside the normalized window coordinate space
                float const aspectRatio = (float) mEgl->getHeight() / mEgl->getWidth();
                float const vertexPosition[] = { aspectRatio, -1.0f, -aspectRatio, -1.0f,
                                                 aspectRatio, 1.0f, -aspectRatio, 1.0f
                                               };

                // Per vertex u and v texture coordinates
                float const textureCoord[] = { 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
                                               0.0f
                                             };

                // Setup fragment program uniforms
                int program = mPlainTextureProgram;
                glUseProgram(program);

                // Setup uniforms
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);
                glUniform1i(glGetUniformLocation(program, "uSourceTex"), 0);

                // Rendering quad
                int aPosCoord = glGetAttribLocation(program, "aPosition");
                int aTexCoord = glGetAttribLocation(program, "aTexCoord");

                glVertexAttribPointer(aPosCoord, 2, GL_FLOAT, GL_FALSE, 0, vertexPosition);
                glVertexAttribPointer(aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, textureCoord);
                glEnableVertexAttribArray(aPosCoord);
                glEnableVertexAttribArray(aTexCoord);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                glDisableVertexAttribArray(aPosCoord);
                glDisableVertexAttribArray(aTexCoord);
                break;
            }

        case EXIT:
            delete [] mProcessedData;
            // NOP
            break;
    }

    mEgl->swap();
}

/**
 *  returns current Sate of the application
 */
NativeApp::State NativeApp::getState(void) const
{
    return mCurrentApplicationState;
}

/**
 * Handle commands.
 */
void NativeApp::handleCommand(int cmd)
{
    switch (cmd)
    {
        case APP_CMD_INIT_WINDOW:
        case APP_CMD_WINDOW_RESIZED:
            mEgl->setWindow(mNativeAppInstance->window);
            break;

        case APP_CMD_TERM_WINDOW:
            mEgl->setWindow(NULL);
            break;
    }
}

/**
 * Handle inputs.
 */
int NativeApp::handleInput(AInputEvent const *event)
{
    // Here we only handle key back event
    int32_t type = AInputEvent_getType(event);

    if (type == AINPUT_EVENT_TYPE_KEY)
    {
        if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK)
        {
            ANativeActivity_finish(mNativeAppInstance->activity);
            return 1;
        }
    }
    else
    {
        int32_t action = AMotionEvent_getAction(event)
                         & AMOTION_EVENT_ACTION_MASK;

        switch (action)
        {
            case AMOTION_EVENT_ACTION_DOWN:
                mScreenPressed = true;
                break;

            case AMOTION_EVENT_ACTION_UP:
                mScreenPressed = false;
                break;
        }

        return 1;
    }

    return 0;
}

/**
 * Wrapper to handle commands generated by the UI.
 */
void NativeApp::HandleCommand(android_app *app, int32_t cmd)
{
    static_cast<NativeApp *>(app->userData)->handleCommand(cmd);
}

/**
 * Wrapper to handle input events generated by the UI.
 */
int32_t NativeApp::HandleInput(android_app *app, AInputEvent *event)
{
    return static_cast<NativeApp *>(app->userData)->handleInput(event);
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(android_app *androidApp)
{
    // Make sure glue isn't stripped.
    app_dummy();

    NvEGLUtil *egl = NvEGLUtil::create();

    if (egl == 0)
    {
        // If we have a basic EGL failure, we need to exit immediately; nothing else we can do
        nv_app_force_quit_no_cleanup(androidApp);
        return;
    }

    NativeApp *instance = new NativeApp(androidApp, egl);

    while (nv_app_status_running(androidApp))
    {
        // Read all pending events.
        int ident, events;
        struct android_poll_source *source;

        // If not rendering, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident = ALooper_pollAll(
                            (nv_app_status_focused(androidApp) ? 1 : 250), NULL, &events,
                            (void **) &source)) >= 0)
        {
            // If we timed out, then there are no pending messages.
            if (ident == ALOOPER_POLL_TIMEOUT)
            {
                break;
            }

            // Process this event.
            if (source != NULL)
            {
                source->process(androidApp, source);
            }

            // Check if we are exiting.  If so, dump out.
            if (!nv_app_status_running(androidApp))
            {
                break;
            }
        }

        if (nv_app_status_interactable(androidApp))
        {
            instance->renderFrame();
        }
    }

    // Remove application instance
    delete instance;
    // Remove EGL instance
    delete egl;
}
