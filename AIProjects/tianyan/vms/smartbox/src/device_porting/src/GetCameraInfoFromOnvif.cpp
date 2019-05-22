#include "string.h"
#include "stdlib.h"
#include "GetCameraInfoFromOnvif.h"
#include "http_client.h"
#include "common_def.h"
#include "cJSON.h"


int ParseJsonToMediaProfile_Inner(cJSON*  pJson, S_Camera_Media_Profile* pMediaProfile)
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

int ParseJsonFromOnvif(char*  pOnvifJson, S_Camera_Info*  pCameraInfoArray, int iAraryMaxSize, int*  piCamCount)
{
	cJSON*   pJson = NULL;
	cJSON*   pArrayDevs = NULL;
	cJSON*   pDev = NULL;
	cJSON*   pProfile = NULL;
	cJSON*   pCountNode = NULL;
	cJSON*   pArrayProfile = NULL;
	int      iRet = 0;
	int      iCount = 0;
	int      iDeviceCount = 0;
	int      iIndex = 0;
	int      iIndex2 = 0;
	int      iProfileCount = 0;

	do 
	{
		pJson = cJSON_Parse((const char *)pOnvifJson);
		BREAK_IN_NULL_POINTER(pJson, iRet, ERR_INVALID_PARAMETER);

		pCountNode = pJson->child;
		iCount = pCountNode->valueint;

		pArrayDevs = pCountNode->next;
		iDeviceCount = cJSON_GetArraySize(pArrayDevs);
		for (iIndex = 0; iIndex < iDeviceCount && iIndex < iAraryMaxSize; iIndex++)
		{
			pDev = cJSON_GetArrayItem(pArrayDevs, iIndex);
			if (pDev != NULL)
			{
				strcpy(pCameraInfoArray[iIndex].strManufacturer, cJSON_GetObjectItem(pDev, "Manufacturer")->valuestring);
				strcpy(pCameraInfoArray[iIndex].strSN, cJSON_GetObjectItem(pDev, "SerialNumber")->valuestring);
				strcpy(pCameraInfoArray[iIndex].strFirmwareVersion, cJSON_GetObjectItem(pDev, "FirmwareVersion")->valuestring);
				strcpy(pCameraInfoArray[iIndex].strHardwareId, cJSON_GetObjectItem(pDev, "HardwareId")->valuestring);
				strcpy(pCameraInfoArray[iIndex].strModel, cJSON_GetObjectItem(pDev, "Model")->valuestring);

				pArrayProfile = cJSON_GetObjectItem(pDev, "Media Profiles");
				iProfileCount = cJSON_GetArraySize(pArrayProfile);
				for (iIndex2 = 0; iIndex2 < iProfileCount && iIndex2 < MAX_PROFILE_COUNT; iIndex2++)
				{
					pProfile = cJSON_GetArrayItem(pArrayProfile, iIndex2);
					ParseJsonToMediaProfile_Inner(pProfile, &(pCameraInfoArray[iIndex].aMediaProfiles[iIndex2]));
				}

				pCameraInfoArray->iProfileCount = iIndex2;
			}
		}

		*piCamCount = iIndex;
	} while (0);

	return iRet;
}


int GetCamerasInfoFromOnvif(char*  pOnvifDiscoverURL, S_Camera_Info*  pCameraInfoArray, int iAraryMaxSize, int*  piCamCount)
{
	int iRet = -1;
	int iBufMaxSize = 0;
	int iJsonDataSize = 0;
	unsigned char* pJsonData = NULL;
	do 
	{
		pJsonData = (unsigned char*)malloc(64 * 1024);
		if (pJsonData == NULL)
		{
			return ERR_LACK_MEMORY;
		}

		iBufMaxSize = 64 * 1024;

		if (pOnvifDiscoverURL == NULL || strlen(pOnvifDiscoverURL) == 0)
		{
			break;
		}

		iRet = HttpGet(pOnvifDiscoverURL, &pJsonData, &iBufMaxSize, &iJsonDataSize);
		if (iRet != 0)
		{
			break;
		}


		ParseJsonFromOnvif((char*)pJsonData, pCameraInfoArray, iAraryMaxSize, piCamCount);

	} while (0);

	if (pJsonData != NULL)
	{
		free(pJsonData);
	}

	return iRet;
}