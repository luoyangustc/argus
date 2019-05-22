#include "MediaConv.h"
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "SmartBox_porting.h"
#include "Onvif_implement.h"
#include "common_def.h"
#include "log4z.h"

int InitSmartBox(S_SmartBox_Info*  pSmartBox, char* pDevcieID, char*  pDeviceDesc)
{
	int iRet = 0;
	int iLen = 0;
	int iStrLen = 0;

	do 
	{
		if (pSmartBox == NULL || pDevcieID == NULL || pDeviceDesc == NULL)
		{
			iRet = -1;
			break;
		}

		memset(pSmartBox, 0, sizeof(S_SmartBox_Info));

		//Do Init the QnSdk  Instance
		//
		//Do Init the QnSdk  Instance
		
		iStrLen = strlen(pDevcieID);
		iLen = iStrLen > (MAX_DEVICE_ID_LENGTH) ? (MAX_DEVICE_ID_LENGTH - 1) : iStrLen;
		memcpy(pSmartBox->strDeviceID, pDevcieID, iLen);

		iStrLen = strlen(pDeviceDesc);
		iLen = iStrLen > (MAX_DEVCIE_DESC_LENGTH) ? (MAX_DEVCIE_DESC_LENGTH - 1) : iStrLen;
		memcpy(pSmartBox->strDeviceDesc, pDeviceDesc, iLen);

	} while (0);

	return iRet;
}

int AddChannelNodeToBox(S_SmartBox_Info*   pSmartBox, S_Channel_Map_Info*  pChannelNodeNew, int iChannelIdx)
{
	int iRet = 0;
	int iIndex = 0;
	S_Channel_Map_Info*   pChannelNode = NULL;

	do
	{

		if (pSmartBox == NULL || pChannelNodeNew == NULL )
		{
			iRet = -1;
			break;
		}

		if (pChannelNodeNew->iChannelNodeType == CM_NODE_ONVIF)
		{
			if ((strlen(pChannelNodeNew->sOnvifInfo.strIP) == 0) || (strlen(pChannelNodeNew->sOnvifInfo.strUser) == 0))
			{
				iRet = -1;
				break;
			}
		}

		if (pChannelNodeNew->iChannelNodeType == CM_NODE_STREAM_URL)
		{
			if (strlen(pChannelNodeNew->sStreamInfo.strStreamURL) == 0)
			{
				iRet = -1;
				break;
			}
		}

		iIndex = FindCurrentChannelNode(pSmartBox, pChannelNodeNew);
		if (iIndex != -1)
		{
			printf("Channel Node, IP:%s, port:%d, Desc:%s already exist!\n", pChannelNodeNew->sOnvifInfo.strIP, pChannelNodeNew->uPort, pChannelNodeNew->strNodeDesc);
			iRet = ERR_ALREADY_EXSIT;
			break;
		}

		pChannelNode = (S_Channel_Map_Info*)malloc(sizeof(S_Channel_Map_Info));
		if (pChannelNode == NULL)
		{
			iRet = -2;
			break;
		}

		memset(pChannelNode, 0, sizeof(S_Channel_Map_Info));

		pSmartBox->pChannelArray[iChannelIdx] = pChannelNode;
		memcpy(pChannelNode, pChannelNodeNew, sizeof(S_Channel_Map_Info));
		iRet = 0;
	} while (0);

	return iRet;
}

int FindCurrentChannelNode(S_SmartBox_Info*   pSmartBox, S_Channel_Map_Info*  pChannelNodeInfo)
{
	int iRet = -1;
	int iIndex = 0;

	do
	{
		if (pSmartBox == NULL || pChannelNodeInfo)
		{
			iRet = -1;
			break;
		}

		for (iIndex = 0; iIndex < (pSmartBox->iChannelCount+1); iIndex++)
		{
			if (pSmartBox->pChannelArray[iIndex] != NULL && pChannelNodeInfo->iChannelNodeType == pSmartBox->pChannelArray[iIndex]->iChannelNodeType)
			{
				switch (pSmartBox->pChannelArray[iIndex]->iChannelNodeType)
				{
					case CM_NODE_ONVIF:
					{
						if (strcmp(pSmartBox->pChannelArray[iIndex]->sOnvifInfo.strIP, pChannelNodeInfo->sOnvifInfo.strIP) == 0 &&
							strcmp(pSmartBox->pChannelArray[iIndex]->strNodeDesc, pChannelNodeInfo->strNodeDesc) == 0)
						{
							iRet = iIndex;
						}
						break;
					}

					case CM_NODE_STREAM_URL:
					{
						if (strcmp(pSmartBox->pChannelArray[iIndex]->sStreamInfo.strStreamURL, pChannelNodeInfo->sStreamInfo.strStreamURL) == 0)
						{
							iRet = iIndex;
						}
						break;
					}

					default:
					{
						break;
					}
				}
			}

			if (iRet != -1)
			{
				break;
			}
		}
	} while (0);

	return iRet;
}

int CheckChannelNodeState(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx)
{
	int iRet = 0;
	S_Channel_Map_Info*   pChannelNode = NULL;

	do
	{
		if (pSmartBox == NULL)
		{
			iRet = -1;
			break;
		}

		if (iChannlNodeIdx >= MAX_CHANNEL_NODE_COUNT || pSmartBox->pChannelArray[iChannlNodeIdx] == NULL)
		{
			iRet = -1;
			break;
		}

		pChannelNode = pSmartBox->pChannelArray[iChannlNodeIdx];

		switch (pChannelNode->iChannelNodeType)
		{
			case CM_NODE_ONVIF:
			{
				iRet = GetDeviceInfoByOnvif(pChannelNode);
				if (iRet == 0)
				{
					pChannelNode->ulChannelNodeState |= (1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS);
					pChannelNode->ulChannelNodeState |= (1 << E_NODE_STATE_CHANNEL_ENABLE);
				}
				break;
			}

			case CM_NODE_STREAM_URL:
			{
				iRet = PeekStreamURL(pChannelNode->sStreamInfo.strStreamURL, &(pChannelNode->sStreamInfo));
				if (iRet == 0)
				{
					pChannelNode->ulChannelNodeState |= (1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS);
				}
				break;
			}
			case CM_NODE_SDK:
			{

				break;
			}
		}
	} while (0);

	return iRet;
}

int CallChannelNodeCaps(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx, int iCapFuncIndex)
{
	int iRet = 0;

	do
	{

	} while (0);

	return iRet;
}

int PushChannelNodeData(S_Channel_Map_Info*  pChannelNode, int iCapFuncIndex, void*  pChannelNodeData)
{
	int iRet = 0;

	do
	{
	} while (0);

	return iRet;
}

int SelectChannelNodeProfileIndex(S_SmartBox_Info*  pSmartBox, int iChannlNodeIdx, int iBandWidthLimit, int iWidthLimit, int iHeightLimit)
{
	int iRet = 0;
	int iBestIndex = -1;
	S_Channel_Map_Info*    pChannelMapInfo = NULL;
	int iIndex = 0;
	int iCurBandWidth = 0;
	int iCurWidth = 0;
	int iCurHeight = 0;


	do 
	{
		if (pSmartBox == NULL || iChannlNodeIdx < 0 || iChannlNodeIdx >= MAX_CHANNEL_NODE_COUNT)
		{
			iRet = -1;
			break;
		}

		pChannelMapInfo = (pSmartBox->pChannelArray[iChannlNodeIdx]);
		if (pChannelMapInfo == NULL)
		{
			break;
		}

		if (pChannelMapInfo->iChannelNodeType == CM_NODE_ONVIF)
		{
			for (iIndex = 0; iIndex < pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount; iIndex++)
			{
				if (pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iBitrate <= iBandWidthLimit &&
					pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iWidth <= iWidthLimit &&
					pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iHeight <= iHeightLimit)
				{
					if (iBestIndex == -1)
					{
						iBestIndex = iIndex;
					}
					else
					{
						iCurBandWidth = pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iBestIndex].iBitrate;
						iCurWidth = pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iBestIndex].iWidth;
						iCurHeight = pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iBestIndex].iHeight;
						if (pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iBitrate >= iCurBandWidth &&
							pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iWidth >= iCurWidth &&
							pChannelMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIndex].iHeight >= iCurHeight)
						{
							iBestIndex = iIndex;
						}
					}
				}
			}
		}

		if (pChannelMapInfo->iChannelNodeType == CM_NODE_STREAM_URL)
		{
			iBestIndex = 0;
		}


		if (iBestIndex != -1)
		{
			pChannelMapInfo->iActiveProfileIndex = iBestIndex;
		}
	} while (0);

	return iBestIndex;
}


int UnInitSmartBox(S_SmartBox_Info*  pSmartBox)
{
	int iRet = 0;

	do
	{
	} while (0);

	return iRet;
}