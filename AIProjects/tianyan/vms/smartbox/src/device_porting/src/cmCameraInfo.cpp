#include "cmCameraInfo.h"
#include "cJSON.h"
#include "string.h"
#include "common_def.h"



int ParseJsonToMediaProfile(cJSON*  pJson, S_Camera_Media_Profile* pMediaProfile)
{
	int iRet = -1;

	do
	{
		if (pJson == NULL || pMediaProfile == NULL)
		{
			break;
		}

		pMediaProfile->iWidth = cJSON_GetObjectItem(pJson, "Width")->valueint;
		pMediaProfile->iHeight = cJSON_GetObjectItem(pJson, "Height")->valueint;
		pMediaProfile->iBitrateInkb = cJSON_GetObjectItem(pJson, "Bitrate(kb)")->valueint;
		strcpy(pMediaProfile->strCodec, cJSON_GetObjectItem(pJson, "Video Codec:")->valuestring);
		strcpy(pMediaProfile->strRtspURL, cJSON_GetObjectItem(pJson, "rtsp url:")->valuestring);
		strcpy(pMediaProfile->strSnapShotURL, cJSON_GetObjectItem(pJson, "Snapshot url:")->valuestring);
		iRet = 0;
	} while (0);

	return iRet;
}


int PasseCamInfoFromJson(char*  pDeviceInfoJson, S_Camera_Info* pCamInfo)
{
	int iRet = 0;
	cJSON*   pJson = NULL;
	cJSON*   pProfile = NULL;
	cJSON*   pArrayProfile = NULL;
	int      iProfileCount = 0;
	int      iIndex = 0;

	do 
	{
		pJson = cJSON_Parse((const char *)pDeviceInfoJson);
		BREAK_IN_NULL_POINTER(pJson, iRet, ERR_INVALID_PARAMETER);

		strcpy(pCamInfo->strManufacturer, cJSON_GetObjectItem(pJson, "Manufacturer")->valuestring);
		strcpy(pCamInfo->strSN, cJSON_GetObjectItem(pJson, "SerialNumber")->valuestring);
		strcpy(pCamInfo->strFirmwareVersion, cJSON_GetObjectItem(pJson, "FirmwareVersion")->valuestring);
		strcpy(pCamInfo->strHardwareId, cJSON_GetObjectItem(pJson, "HardwareId")->valuestring);
		strcpy(pCamInfo->strModel, cJSON_GetObjectItem(pJson, "Model")->valuestring);

		pArrayProfile = cJSON_GetObjectItem(pJson, "Media Profiles");
		iProfileCount = cJSON_GetArraySize(pArrayProfile);
		for (iIndex = 0; iIndex < iProfileCount && iIndex < MAX_PROFILE_COUNT; iIndex++)
		{
			pProfile = cJSON_GetArrayItem(pArrayProfile, iIndex);
			ParseJsonToMediaProfile(pProfile, &(pCamInfo->aMediaProfiles[iIndex]));
		}

	} while (0);

	if (pJson != NULL)
	{
		cJSON_Delete(pJson);
	}

	return iRet;
}