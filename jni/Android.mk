LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := lib_l3
LOCAL_SRC_FILES := ../cuda/lib_l3.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := lib_bilateralfilter
LOCAL_SRC_FILES := ../cuda/lib_bilateralfilter.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libcudart_static
LOCAL_LIB_PATH   += $(CUDA_TOOLKIT_ROOT)/targets/armv7-linux-androideabi/lib/
LOCAL_SRC_FILES  := $(LOCAL_LIB_PATH)/libcudart_static.a 
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := bilateral_filter

NVPACK := $(NDK_ROOT)/..
BILATERAL_FILTER_ROOT := $(LOCAL_PATH)/..

MY_PREFIX       := $(LOCAL_PATH)/
MY_SOURCES      := $(wildcard $(LOCAL_PATH)/*.cpp)
LOCAL_SRC_FILES := $(MY_SOURCES:$(MY_PREFIX)%=%)
 
LOCAL_STATIC_LIBRARIES := lib_bilateralfilter lib_l3 libcudart_static
LOCAL_STATIC_LIBRARIES += nv_and_util nv_egl_util nv_glesutil nv_shader nv_file
LOCAL_LDLIBS := -llog -landroid -lGLESv2 -lEGL 
LOCAL_C_INCLUDES += $(BILATERAL_FILTER_ROOT)/cuda
LOCAL_C_INCLUDES += $(CUDA_TOOLKIT_ROOT)/targets/armv7-linux-androideabi/include

include $(BUILD_SHARED_LIBRARY)

$(call import-add-path, $(NVPACK)/Samples/TDK_Samples/libs/jni)

$(call import-module,nv_and_util)
$(call import-module,nv_egl_util)
$(call import-module,nv_shader)
$(call import-module,nv_file)
$(call import-module,nv_glesutil)
