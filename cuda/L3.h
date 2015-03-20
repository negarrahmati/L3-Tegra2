/*
 * L3.h
 *
 *  Created on: Mar 12, 2015
 *      Author: mvc
 */

#ifndef L3_H_
#define L3_H_
#include <android/log.h>

#define APP_NAME "CUDA_NEGAR"

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_ERROR,  \
                                             APP_NAME, \
                                             __VA_ARGS__))
class L3 {
public:
	L3();
	virtual ~L3();

	int L3_main(void);


};

#endif /* L3_H_ */
