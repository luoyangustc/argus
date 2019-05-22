#ifndef __GET_CAMERA_INFO_FROM_CFG_H__
#define __GET_CAMERA_INFO_FROM_CFG_H__

#include "cmCameraInfo.h"

int GetCamerasInfoFromOnvif(char*  pOnvifDiscoverURL, S_Camera_Info*  pCameraInfoArray, int iAraryMaxSize, int*  piCamCount);

#endif