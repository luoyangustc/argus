#ifndef __SMART_BOX_PORTING_H__
#define __SMART_BOX_PORTING_H__

#include "gsoap_common.h"


#define  MAX_DEVICE_ID_LENGTH					64
#define  MAX_DEVCIE_DESC_LENGTH					512
#define  MAX_CHANNEL_NODE_COUNT					256
#define  MAX_CHANNEL_NODE_DESC_MAX_LENGTH       512


#define CM_NODE_ONVIF         0
#define CM_NODE_STREAM_URL    1
#define CM_NODE_SDK           2

typedef  enum
{
	E_NODE_STATE_CHANNEL_ENABLE = 0,
	E_NODE_STATE_LOCAL_ONLINE_FLAG_POS = 1,
	E_NODE_STATE_REMOTE_ONLIEN_FLAG_POS = 2,
	E_NODE_STATE_REMOTE_DATA_TRANSFER_FLAG_POS = 3,
}E_Node_State_POS;

typedef  enum
{
	E_NODE_FUNC_VIDEO_FLAG_POS = 0,
	E_NODE_FUNC_SNAPSHOT_FLAG_POS = 1,
	E_NODE_FUNC_PTZ_FLAG_POS = 2,
	E_NODE_FUNC_EVENT_FLAG_POS = 3,
	E_NODE_FUNC_CMD_PARAM_GET = 4,	
	E_NODE_FUNC_CMD_PARAM_SET = 5,
	E_NODE_FUNC_MAX_FLAG_POS = 32,
}E_Channel_Node_Func_Pos;


typedef struct
{
	char   strIP[128];
	char   strUser[128];
	char   strPwd[128];
	S_Dev_Info    sDevOnvifInfo;
}  S_ONVIF_ITEM_INFO;

typedef struct
{
	char                strStreamURL[1024];
	int                 iVideoCodec;
	int                 iWidth;
	int                 iHeight;
	unsigned char       uVideoExtra[128];
	int                 iVideoExtraSize;
	int                 iAudioCodec;
	int                 iAudioSampleRate;
	int                 iAudioChannel;
	int                 iAudioBitWidth;
	unsigned char       uAudioExtra[128];
	int                 iAudioExtraSize;
} S_STREAM_ITEM_INFO;


typedef struct
{
	union
	{
		S_ONVIF_ITEM_INFO   sOnvifInfo;
		S_STREAM_ITEM_INFO  sStreamInfo;
	};

	short  uPort;
	char   strNodeDesc[MAX_CHANNEL_NODE_DESC_MAX_LENGTH];
	int    iChannelNodeType;
	unsigned int ulChannelNodeCapability;
	char*        pCapabilityDescArray[E_NODE_FUNC_MAX_FLAG_POS];
	unsigned int ulChannelNodeState;
	int          iActiveProfileIndex;
	long long    ullLastActiveTime;
}S_Channel_Map_Info;

typedef struct
{
	char   strDeviceID[MAX_DEVICE_ID_LENGTH];
	char   strDeviceDesc[MAX_DEVCIE_DESC_LENGTH];
	int    iChannelCount;
	S_Channel_Map_Info*   pChannelArray[MAX_CHANNEL_NODE_COUNT];
	void*   pQnSdkIns;
	unsigned int ulDeviceState;
	int      iMaxChannelNodeCount;
}S_SmartBox_Info;


int InitSmartBox(S_SmartBox_Info*  pSmartBox, char* pDevcieID, char*  pDeviceDesc);
int AddChannelNodeToBox(S_SmartBox_Info*   pSmartBox, S_Channel_Map_Info*  pChannelNodeInfo, int iChannelIdx);
int FindCurrentChannelNode(S_SmartBox_Info*   pSmartBox, S_Channel_Map_Info*  pChannelNodeInfo);
int CheckChannelNodeState(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx);
int CallChannelNodeCaps(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx, int iCapFuncIndex);
int PushChannelNodeData(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx, int iCapFuncIndex, void*  pChannelNodeData);
int UnInitSmartBox(S_SmartBox_Info*  pSmartBox);
int SelectChannelNodeProfileIndex(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx, int iBandWidthLimit, int iWidthLimit, int iHeightLimit);

#endif