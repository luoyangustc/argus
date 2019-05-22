
#include "MediaConv.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "SmartBox_porting.h"
#include "GetDeviceInfoFromBZSetting.h"
#include "Onvif_implement.h"
#include "common_def.h"
#include "qiniu_dev_net_porting.h"
#include "LoadCfg.h"

#ifdef _LINUX_
#include <unistd.h>
#endif

#ifdef _WIN32
#include "windows.h"
#endif


#include "log4z.h"


int ParseCmd(int iArgc, char*  pArgv[], char**  ppSmartBoxCfgFilePath);

int main(int argc, char* argv[])
{
	zsummer::log4z::ILog4zManager::getRef().start();

	Dev_Cmd_Param_t   sCmd = {0};
	int iRet = NULL;
	char*  pSmarBoxCfgPath = NULL;
	int iDeviceCount = 0;
	int iIndex = 0;
	S_SDK_Ins   sSdkIns;
	S_Channel_Map_Info*  pChannelNodeArray = NULL;
	S_SmartBox_Info   sSmartBox;
	char            strSmartBoxBZSettingURL[1024] = { 0 };
	char            strSmartBoxBZBaseURL[1024] = { 0 };
	char            strSmartChannelURL[1024] = { 0 };
	char            strSmartBoxDeviceId[1024] = { 0 };
	char            strEntryServerIp[1024] = { 0 };
	int             iEntryServerPort =  0 ;
	char strInputMediaURL[1024] = { 0 };
	char strSnapshotURL[1024] = { 0 };
	S_Stream_CTX*    pStreamCtx = NULL;
	int             iMaxChannelNodeCount = 0;
	int             iStreamIdx = 0;

	do
	{
		iRet = ParseCmd(argc, argv, &pSmarBoxCfgPath);
		if (iRet != 0)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		iRet = LoadCfg(pSmarBoxCfgPath, strSmartBoxBZBaseURL, strSmartBoxBZSettingURL, strSmartChannelURL, strSmartBoxDeviceId, strEntryServerIp, &iEntryServerPort);
		if (iRet != 0)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		memset(&sSmartBox, 0, sizeof(S_SmartBox_Info));
		pChannelNodeArray = (S_Channel_Map_Info*)malloc(sizeof(S_Channel_Map_Info) * 256);
		if (pChannelNodeArray == NULL)
		{
			iRet = ERR_LACK_MEMORY;
			break;
		}

		memset(pChannelNodeArray, 0, sizeof(S_Channel_Map_Info) * 256);
		//Get CamInfo From BZ
		iRet = GetDeviceInfoFromBZSetting(strSmartBoxBZBaseURL, strSmartBoxBZSettingURL, strSmartBoxDeviceId, pChannelNodeArray, 256, &iDeviceCount, &iMaxChannelNodeCount);
		if (iRet == 0)
		{
			for (iIndex = 0; iIndex < iMaxChannelNodeCount; iIndex++)
			{
				iRet = AddChannelNodeToBox(&sSmartBox, &(pChannelNodeArray[iIndex]), iIndex);
				if (iRet == 0)
				{
					sSmartBox.pChannelArray[iIndex]->iActiveProfileIndex = -1;
					CheckChannelNodeState(&sSmartBox, iIndex);
				}
			}
		}

		iRet = InitSDK(&sSdkIns, &sSmartBox, NULL, strSmartBoxDeviceId, strEntryServerIp, iEntryServerPort, iMaxChannelNodeCount);
		if (iRet != 0)
		{
			LOGFMTI("InitSDK Error! \n");
			break;
		}

		memset(sSdkIns.strBZChannelInfoReqURL, 0, sizeof(sSdkIns.strBZChannelInfoReqURL));
		strcpy(sSdkIns.strBZChannelInfoReqURL, strSmartChannelURL);
		//Select the right media Profile
		for (iIndex = 0; iIndex < sSmartBox.iMaxChannelNodeCount; iIndex++)
		{
			if (sSmartBox.pChannelArray[iIndex] != NULL && (sSmartBox.pChannelArray[iIndex]->ulChannelNodeState & ((1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS) | (1 << E_NODE_STATE_CHANNEL_ENABLE))))
			{
				//Select the Stream idx first
				iStreamIdx = SelectChannelNodeProfileIndex(&sSmartBox, iIndex, 1024, 1280, 720);
				if (iStreamIdx != -1)
				{
					//For Convenient Index, the server begin index from 1
					AddStreamCtxSDKIns(&sSdkIns, iIndex, iStreamIdx);
					if (iStreamIdx != -1)
					{
						LOGFMTI("the select rtsp url:%s\n", sSmartBox.pChannelArray[iIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIdx].strMainStreamRtspURL);
						LOGFMTI("the select snapshot url:%s\n", sSmartBox.pChannelArray[iIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIdx].strMainStreamSnapshotURL);
					}
				}
			}
		}

		for (iIndex = 0; iIndex < sSmartBox.iMaxChannelNodeCount; iIndex++)
		{
			if (sSmartBox.pChannelArray[iIndex] != NULL)
			{
				for (iStreamIdx = 0; iStreamIdx < MAX_SMARTBOX_STREAM_COUNT; iStreamIdx++)
				{
					if (sSdkIns.pStreamCtxArray[iIndex][iStreamIdx] != NULL)
					{
						//iRet = StartMediaPush(&sSdkIns, iIndex, NULL, NULL, iStreamIdx);
					}
				}
			}
		}


		while (1)
		{
			memset(&sCmd, 0, sizeof(Dev_Cmd_Param_t));
			iRet = GetCmdItem(&sSdkIns, &sCmd);
			if (iRet == 0)
			{
				HandleCmd(&sSdkIns, &sCmd);
			}

#ifdef _LINUX_
			usleep(5000);
#endif

#ifdef _WIN32
			Sleep(5);
#endif
		}


	} while (0);

	return 0;
}



int ParseCmd(int iArgc, char*  pArgv[], char**  ppSmartBoxCfgFilePath)
{
	int  iIndex = 0;
	float    fRate = 0;
	int     iRet = 0;
	char*   pCurOpt = NULL;
	char*   pCurValue = NULL;


	if (iArgc < 2)
	{
		LOGFMTI("invalid parameter!\n");
		LOGFMTI("-dev_info device_onvif_discover_url    set the device info interface\n"
			);
		return -1;
	}

	while (iIndex < iArgc)
	{
		pCurOpt = pArgv[iIndex];
		if (strcmp("-smartbox_cfg", pCurOpt) == 0)
		{
			*ppSmartBoxCfgFilePath = pArgv[++iIndex];
		}

		iIndex++;
	}

	return 0;
}