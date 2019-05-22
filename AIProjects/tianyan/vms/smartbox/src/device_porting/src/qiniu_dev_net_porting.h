#ifndef __QINIU_DEV_NET_PORTING_H__
#define __QINIU_DEV_NET_PORTING_H__


#include "DeviceSDK.h"
#include "SmartBox_porting.h"
#include "pthread.h"

#define SDK_STATUS_NONE 0
#define SDK_STATUS_INITED 1
#define SDK_STATUS_OPEN_MEDIA 2
#define SDK_INITED_UNINITED 3


#define MEDIA_VIDEO_TYPE 0
#define MEDIA_AUDIO_TYPE 1
#define MEDIA_INFO_TYPE  2

#define FRM_VIDEO_P_TYPE 0
#define FRM_VIDEO_I_TYPE 1

#define MAX_SMARTBOX_DEVICE_COUNT 256
#define MAX_SMARTBOX_STREAM_COUNT 4
#define MAX_SMARTBOX_CACHE_MSG_COUNT 1024

typedef struct
{
	int             iFrmBufNum;
	unsigned int    ulAudioTimestamp;
	unsigned int    ulTotalFrmNum;
	unsigned int    ulVideoFrmNum;
	unsigned int    ulAudioFrmNum;
	unsigned char   uStatus;//×´Ì¬0¹Ø±Õ1´ò¿ª
	unsigned char*     pImageBuffer;
	int                iImageBufMaxSize;
	int                iImageDataSize;
	unsigned char*     pFrameBuffer;
	int                iFrameBufMaxSize;
	int                iFrameDataSize;
	int                iVideoHeaderParsed;
	int                iAudioHeaderParsed;
	S_Media_Conv       sMediaConvIns;
	pthread_t          hPthreadHandle;
	long               lLastActiveTime;
} S_Stream_CTX;


typedef struct
{
	Dev_Info_t   sDevInfo;
	S_SmartBox_Info*   pSmartBoxInfo;
	S_Stream_CTX*      pStreamCtxArray[MAX_SMARTBOX_DEVICE_COUNT][MAX_SMARTBOX_STREAM_COUNT];
	int                iCurStreamCount;
	pthread_mutex_t    sMutex;
	Dev_Cmd_Param_t*   pDevCmd[MAX_SMARTBOX_CACHE_MSG_COUNT];
	int                iCurCmdReadIdx;
	int                iCurCmdWritedx;
	int                iCmdCount;
	char               strBZChannelInfoReqURL[1024];
} S_SDK_Ins;

typedef struct
{
	unsigned char*   pFrameData;
	int iFrameSize;
	int iMediaType;
	int iMediaCodecType;
	int iFrameType;
	unsigned int ulTimeStamp;
} S_Frame;


void SdkAppCallback(void* user_data, Dev_Cmd_Param_t *args);
int InitSDK(S_SDK_Ins*  pSdkIns, S_SmartBox_Info*  pSmartBoxInfo, char*  pDevCfg, char* pDeviceId, char* pEntryServerIp, int iEntryServerPort, int iMaxChannelNodeCount);
int GetCmdItem(S_SDK_Ins*  pSdkIns, Dev_Cmd_Param_t *args);
int UnInitSDK(S_SDK_Ins*  pSdkIns);
int StartMediaPush(S_SDK_Ins*  pSdkIns, int iChannelIndex, char* pOutuptURL, char* pOutputFmt, int iStreamIndex);
int StopMediaPush(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIndex);
int GetCurMediaURLS(S_SDK_Ins*  pSdkIns, int iChannelIndex, char* pRtspMediaURL, char*  pSnapShotURL, int iStreamIndex);
int AddStreamCtxSDKIns(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIndex);
int SdkStreamInfoReport(S_SDK_Ins*  pSdkIns, S_Frame*   pFrame, int iChannelIndex, int iStreamIndex);
int SmartBoxChannelInfoConv(S_SDK_Ins*  pSdkIns, S_SmartBox_Info*  pSmartBoxInfo);
int TransactMediaFrameFromFFmpeg(void*  pFrameFromFFmpeg, void* pMediaConvCtx, void*  pSdkIns, int iChannelIndex, int iStreamIndex);
int HandleCmd(S_SDK_Ins*  pSdkIns, Dev_Cmd_Param_t *args);
int DelChannelFromSmartBox(S_SDK_Ins*  pSdkIns, int iIndex);

#endif __QINIU_DEV_NET_PORTING_H__
