#include "Onvif_implement.h"
#include "gsoap_common.h"


int GetDeviceInfoByOnvif(S_Channel_Map_Info*  pChannelMapInfo)
{
	int iRet = 0;
	char  strBaseDeviceServURL[256] = { 0 };
	S_Dev_Info   sDevInfo;

	do 
	{
		if (pChannelMapInfo == NULL)
		{
			iRet = 1;
			break;
		}

		memset(&sDevInfo, 0, sizeof(S_Dev_Info));
		sprintf(strBaseDeviceServURL, "http://%s/onvif/device_service", pChannelMapInfo->sOnvifInfo.strIP);
		iRet = OnvifGetDeviceBasicInfo(pChannelMapInfo->sOnvifInfo.strUser, pChannelMapInfo->sOnvifInfo.strPwd, strBaseDeviceServURL, &(pChannelMapInfo->sOnvifInfo.sDevOnvifInfo));
		if (iRet != 0)
		{
			printf("Onvif GetDeviceBasicInfo Error!, onvif req url:%s\n", strBaseDeviceServURL);
			break;
		}

		iRet = OnvifGetCapabilities(pChannelMapInfo->sOnvifInfo.strUser, pChannelMapInfo->sOnvifInfo.strPwd, strBaseDeviceServURL, &(pChannelMapInfo->sOnvifInfo.sDevOnvifInfo));
		if (iRet != 0)
		{
			printf("Onvif GetCapabilities Error!\n");
			break;
		}

		iRet = OnvifGetProfiles(pChannelMapInfo->sOnvifInfo.strUser, pChannelMapInfo->sOnvifInfo.strPwd, pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.strMediaAddr, &(pChannelMapInfo->sOnvifInfo.sDevOnvifInfo));
		if (iRet != 0)
		{
			printf("Onvif GetProfiles Error!\n");
			break;
		}

		iRet = OnvifGetDeviceVideo(pChannelMapInfo->sOnvifInfo.strUser, pChannelMapInfo->sOnvifInfo.strPwd, pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.strMediaAddr, &(pChannelMapInfo->sOnvifInfo.sDevOnvifInfo));
		if (iRet != 0)
		{
			printf("Onvif GetDeviceVideo Error!\n");
			break;
		}

		iRet = OnvifGetDeviceImage(pChannelMapInfo->sOnvifInfo.strUser, pChannelMapInfo->sOnvifInfo.strPwd, pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.strMediaAddr, &(pChannelMapInfo->sOnvifInfo.sDevOnvifInfo));
		if (iRet != 0)
		{
			printf("Onvif GetDeviceImage Error!\n");
			break;
		}

	} while (0);

	return iRet;
}
