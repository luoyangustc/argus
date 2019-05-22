#ifndef __CM_CAMERA_INFO_H__
#define __CM_CAMERA_INFO_H__


#define MAX_PROFILE_COUNT 8

typedef struct
{
	int iWidth;
	int iHeight;
	int iBitrateInkb;
	char   strCodec[64];
	char   strRtspURL[256];
	char   strSnapShotURL[256];
} S_Camera_Media_Profile;


typedef struct
{
	char   strSN[128];
	char   strManufacturer[128];
	char   strModel[128];
	char   strFirmwareVersion[32];
	char   strHardwareId[128];
	int    iProfileCount;
} S_Camera_Info;



int PasseCamInfoFromJson(char*  pDeviceInfoJson, S_Camera_Info* pCamInfo);





#endif