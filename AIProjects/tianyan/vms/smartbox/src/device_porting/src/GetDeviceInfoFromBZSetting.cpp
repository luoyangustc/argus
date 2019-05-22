#include "GetDeviceInfoFromBZSetting.h"
#include "string.h"
#include "http_client.h"
#include "common_def.h"
#include "cJSON.h"

#define SMARTBOX_PROTOCOL_ONVIF 1
#define SMARTBOX_PROTOCOL_STREAM_URL 2

static int ParseSmartBoxBaseInfo(unsigned char*   pData, int iSize, int* pChannelCount);
static int ParseSmartBoxInfo(unsigned char*   pData, int iSize, S_Channel_Map_Info*  pChannelNodeArray, int iArrayMaxSize, int*  pChannelNodeCount, int*  pMaxChannelNodeCount);
static void   AssembleDeviceBaseInfoReqURL(char* pDeviceInfoURL, char* pBaseURL, char* pDeviceID);
static void   AssembleDeviceDetailInfoReqURL(char* pDeviceInfoURL, char* pBaseURL, char* pDeviceID);
static int ParseChannelInfo(unsigned char*   pData, int iSize, S_Channel_Map_Info*  pChannelNode, int iIndex);
static int ParseChannelInfoFromJson(cJSON*   pJsonRoot, S_Channel_Map_Info*  pChannelNode, int iIndex);


int GetDeviceInfoFromBZSetting(char* pBZBaseInterface, char* pBZInterface, char*  pStrSmarboxDeviceId, S_Channel_Map_Info*  pChannelNodeArray, int iArrayMaxSize, int*  pChannelNodeCount, int*  pMaxChannelNodeCount)
{
	char   strDeviceInfoBaseURL[1024] = { 0 };
	char   strDeviceInfoURL[1024] = { 0 };
	int iRet = 0;
	int iChannelCount = 0;
	unsigned char*   pRecBuf = NULL;
	int              iRecMaxSize = NULL;
	int              iRecSize = 0;

	do 
	{

		AssembleDeviceBaseInfoReqURL(strDeviceInfoBaseURL, pBZBaseInterface, pStrSmarboxDeviceId);
		iRet = HttpGet(strDeviceInfoBaseURL, &pRecBuf, &iRecMaxSize, &iRecSize);
		if (iRet != 0)
		{
			break;
		}

		ParseSmartBoxBaseInfo(pRecBuf, iRecSize, &iChannelCount);
		*pMaxChannelNodeCount = iChannelCount;

		AssembleDeviceDetailInfoReqURL(strDeviceInfoURL, pBZInterface, pStrSmarboxDeviceId);
		iRet = HttpGet(strDeviceInfoURL, &pRecBuf, &iRecMaxSize, &iRecSize);
		if (iRet != 0)
		{
			break;
		}

		ParseSmartBoxInfo(pRecBuf, iRecSize, pChannelNodeArray, iArrayMaxSize, pChannelNodeCount, pMaxChannelNodeCount);
	} while (0);

	if (pRecBuf != NULL)
	{
		free(pRecBuf);
	}

	return iRet;
}


int GetChannelInfoFromBZSetting(char* pBZInterface, S_Channel_Map_Info*  pChannelNode, int iIndex)
{
	int iRet = 0;
	unsigned char*   pRecBuf = NULL;
	int              iRecMaxSize = NULL;
	int              iRecSize = 0;

	do 
	{
		iRet = HttpGet(pBZInterface, &pRecBuf, &iRecMaxSize, &iRecSize);
		if (iRet != 0)
		{
			break;
		}


		ParseChannelInfo(pRecBuf, iRecSize, pChannelNode, iIndex);
	} while (0);

	if (pRecBuf != NULL)
	{
		free(pRecBuf);
	}
	return iRet;
}


int ParseSmartBoxInfo(unsigned char*   pData, int iSize, S_Channel_Map_Info*  pChannelNodeArray, int iArrayMaxSize, int*  pChannelNodeCount, int*  pMaxChannelNodeCount)
{
	int iRet = 0;
	cJSON*  pRoot = NULL;
	cJSON*  pJSonData = NULL;
	cJSON*  pNode = NULL;
	cJSON*  pDevices = NULL;
	cJSON*  pCameraItem = NULL;
	cJSON*  pChannelId = NULL;
	cJSON*  pType = NULL;
	cJSON*  pAttribute = NULL;
	S_Channel_Map_Info*   pChannelMapInfo = NULL;

	int     iIdValue = 0;
	int     iValue = 0;
	int     iIndex = 0;
	int     iCount = 0;
	char    srtValue[1024] = { 0 };
	int     iCameraCount = 0;
	do 
	{
		if (pData == NULL)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		pRoot = cJSON_Parse((const char *)pData);
		BREAK_IN_NULL_POINTER(pRoot, iRet, ERR_INVALID_PARAMETER);

		pNode = pRoot->child;
		BREAK_IN_NULL_POINTER(pNode, iRet, ERR_INVALID_PARAMETER);

		BREAK_IN_ERR_NOT_ZERO(pNode->valueint, iRet, ERR_INVALID_PARAMETER);

		//Get the Cameras Array node
		pJSonData = cJSON_GetObjectItem(pRoot, "data");
		BREAK_IN_NULL_POINTER(pJSonData, iRet, ERR_INVALID_PARAMETER);

		//Get the Cameras Array node
		pDevices = cJSON_GetObjectItem(pJSonData, "sub_devices");
		BREAK_IN_NULL_POINTER(pDevices, iRet, ERR_INVALID_PARAMETER);

		iCount = cJSON_GetArraySize(pDevices);
		for (iIndex = 0; iIndex < iCount; iIndex++)
		{
			pCameraItem = cJSON_GetArrayItem(pDevices, iIndex);
			if (pCameraItem != NULL)
			{
				memset(srtValue, 0, 1024);
				pNode = cJSON_GetObjectItem(pCameraItem, "channel_id");
				if (pNode != NULL)
				{
					iIdValue = pNode->valueint;
					if (iIdValue > 0 && iIdValue <= (*pMaxChannelNodeCount))
					{
						pChannelMapInfo = &(pChannelNodeArray[iIdValue - 1]);
						ParseChannelInfoFromJson(pCameraItem, pChannelMapInfo, iIdValue - 1);
					}
					else
					{
						continue;
					}
				}
			}
		}
	} while (0);

	if (pRoot != NULL)
	{
		cJSON_Delete(pRoot);
	}

	*pChannelNodeCount = iCameraCount;
	return iRet;
}

static void   AssembleDeviceDetailInfoReqURL(char* pDeviceInfoURL, char* pBaseURL, char* pDeviceID)
{
	do 
	{
		sprintf(pDeviceInfoURL, pBaseURL, pDeviceID);
	} while (0);

	return;
}

static void   AssembleDeviceBaseInfoReqURL(char* pDeviceInfoURL, char* pBaseURL, char* pDeviceID)
{
	do
	{
		sprintf(pDeviceInfoURL, pBaseURL, pDeviceID);
	} while (0);

	return;
}

static int ParseSmartBoxBaseInfo(unsigned char*   pData, int iSize, int* pChannelCount)
{
	int iRet = 0;
	cJSON*  pRoot = NULL;
	cJSON*  pJSonData = NULL;
	cJSON*  pNode = NULL;
	cJSON*  pChannelNum = NULL;

	do 
	{
		if (pData == NULL)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		pRoot = cJSON_Parse((const char *)pData);
		BREAK_IN_NULL_POINTER(pRoot, iRet, ERR_INVALID_PARAMETER);


		//Get the Cameras Array node
		pJSonData = cJSON_GetObjectItem(pRoot, "data");
		BREAK_IN_NULL_POINTER(pJSonData, iRet, ERR_INVALID_PARAMETER);

		//Get the SmarbBox attribute
		pNode = cJSON_GetObjectItem(pJSonData, "attribute");
		BREAK_IN_NULL_POINTER(pNode, iRet, ERR_INVALID_PARAMETER);

		//Get the SmarbBox attribute
		pChannelNum = cJSON_GetObjectItem(pNode, "channel_num");
		BREAK_IN_NULL_POINTER(pChannelNum, iRet, ERR_INVALID_PARAMETER);

		*pChannelCount = pChannelNum->valueint;
		
	} while (0);

	if (pRoot != NULL)
	{
		cJSON_Delete(pRoot);
	}

	return iRet;
}

static int ParseChannelInfo(unsigned char*   pData, int iSize, S_Channel_Map_Info*  pChannelNode, int iIndex)
{
	int iRet = 0;
	cJSON*  pRoot = NULL;
	cJSON*  pJSonData = NULL;
	cJSON*  pNode = NULL;
	cJSON*  pAttribute = NULL;
	char    srtValue[1024] = { 0 };

	do
	{
		if (pData == NULL)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		pRoot = cJSON_Parse((const char *)pData);
		BREAK_IN_NULL_POINTER(pRoot, iRet, ERR_INVALID_PARAMETER);

		//Get the data Object
		pJSonData = cJSON_GetObjectItem(pRoot, "data");
		BREAK_IN_NULL_POINTER(pJSonData, iRet, ERR_INVALID_PARAMETER);

		iRet = ParseChannelInfoFromJson(pJSonData, pChannelNode, iIndex);
		//Get the channel id
	} while (0);

	if (pRoot != NULL)
	{
		cJSON_Delete(pRoot);
	}

	return iRet;
}


int ParseChannelInfoFromJson(cJSON*   pJsonRoot, S_Channel_Map_Info*  pChannelNode, int iIndex)
{
	int iRet = 0;
	cJSON*   pNode = NULL;
	cJSON*   pAttribute = NULL;

	do 
	{
		pNode = cJSON_GetObjectItem(pJsonRoot, "channel_id");
		BREAK_IN_NULL_POINTER(pNode, iRet, ERR_INVALID_PARAMETER);
		if (pNode->valueint != (iIndex + 1))
		{
			iRet = -1;
			break;
		}

		pNode = cJSON_GetObjectItem(pJsonRoot, "type");
		if (pNode != NULL)
		{
			pChannelNode->iChannelNodeType = pNode->valueint;
		}

		//Get the attribute
		pAttribute = cJSON_GetObjectItem(pJsonRoot, "attribute");
		BREAK_IN_NULL_POINTER(pAttribute, iRet, ERR_INVALID_PARAMETER);

		pNode = cJSON_GetObjectItem(pAttribute, "discovery_protocol");
		if (pNode != NULL)
		{
			if (pNode->valueint == SMARTBOX_PROTOCOL_ONVIF)
			{
				pChannelNode->iChannelNodeType = CM_NODE_ONVIF;
			}

			if (pNode->valueint == SMARTBOX_PROTOCOL_STREAM_URL)
			{
				pChannelNode->iChannelNodeType = CM_NODE_STREAM_URL;
			}
		}

		switch (pChannelNode->iChannelNodeType)
		{
			case CM_NODE_ONVIF:
			{
				pNode = cJSON_GetObjectItem(pAttribute, "ip");
				if (pNode != NULL)
				{
					strcpy(pChannelNode->sOnvifInfo.strIP, pNode->valuestring);
				}

				pNode = cJSON_GetObjectItem(pAttribute, "account");
				if (pNode != NULL)
				{
					strcpy(pChannelNode->sOnvifInfo.strUser, pNode->valuestring);
				}

				pNode = cJSON_GetObjectItem(pAttribute, "password");
				if (pNode != NULL)
				{
					strcpy(pChannelNode->sOnvifInfo.strPwd, pNode->valuestring);
				}
				break;
			}

			case CM_NODE_STREAM_URL:
			{
				pNode = cJSON_GetObjectItem(pAttribute, "upstream_url");
				if (pNode != NULL)
				{
					strcpy(pChannelNode->sStreamInfo.strStreamURL, pNode->valuestring);
				}
				break;
			}
		}


		pNode = cJSON_GetObjectItem(pAttribute, "name");
		if (pNode != NULL)
		{
			strcpy(pChannelNode->strNodeDesc, pNode->valuestring);
		}

		pNode = cJSON_GetObjectItem(pAttribute, "vendor");
		if (pNode != NULL)
		{
			strcat(pChannelNode->strNodeDesc, ";");
			strcat(pChannelNode->strNodeDesc, pNode->valuestring);
		}

	} while (0);

	return iRet;
}