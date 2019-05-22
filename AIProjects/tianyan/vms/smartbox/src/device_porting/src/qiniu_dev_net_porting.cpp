#ifdef OS_LINUX
#include <signal.h>
#endif


#include "MediaConv.h"
#include "qiniu_dev_net_porting.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "CameraSnapShot.h"
#include "UAVParser.h"
#include "common_def.h"
#include "log4z.h"
#include "time.h"
#include "GetDeviceInfoFromBZSetting.h"

//FILE*   _gVideoDataDump[8] = { 0 };
//FILE*   _gVideoInfoDump[8] = { 0 };


#define MIN_LOG_OUTPUT_INTERVAL    120

static int GetSampleRateValue(int iSampleRate)
{
	int iRet = 0;
	switch (iSampleRate)
	{
		case 8000:
		{
			iRet = EN_AUDIO_SR_8_KHZ;
			break;
		}

		case 11025:
		{
			iRet = EN_AUDIO_SR_11_025_KHZ;
			break;
		}
		
		case 12000:
		{
			iRet = EN_AUDIO_SR_12_KHZ;
			break;
		}
	
		case 16000:
		{
			iRet = EN_AUDIO_SR_16_KHZ;
			break;
		}
	
		case 22050:
		{
			iRet = EN_AUDIO_SR_22_05_KHZ;
			break;
		}
	
		case 24000:
		{
			iRet = EN_AUDIO_SR_24_KHZ;
			break;
		}
	
		case 32000:
		{
			iRet = EN_AUDIO_SR_32_KHZ;
			break;
		}

		case 44100:
		{
			iRet = EN_AUDIO_SR_44_1_KHZ;
			break;
		}
	
		case 48000:
		{
			iRet = EN_AUDIO_SR_48_KHZ;
			break;
		}
	
		case 64000:
		{
			iRet = EN_AUDIO_SR_64_KHZ;
			break;
		}
	
		case 88200:
		{
			iRet = EN_AUDIO_SR_88_2_KHZ;
			break;
		}

		case 96000:
		{
			iRet = EN_AUDIO_SR_96_KHZ;
			break;
		}

		default:
		{
			break;
		}
	}

	return iRet;
}

static void ResetCTXForStreamCTX(S_Stream_CTX* pStreamCtx)
{
	pStreamCtx->iFrmBufNum = 0;
	pStreamCtx->ulAudioTimestamp = 0;
	pStreamCtx->ulTotalFrmNum = 0;
	pStreamCtx->ulVideoFrmNum = 0;
	pStreamCtx->ulAudioFrmNum = 0;
	pStreamCtx->uStatus = 0;
	pStreamCtx->iImageDataSize = 0;
	pStreamCtx->iFrameDataSize = 0;
	pStreamCtx->iVideoHeaderParsed = 0;
	pStreamCtx->iAudioHeaderParsed;
	pStreamCtx->lLastActiveTime = 0;
}


static void HandleOpenStream(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIdx)
{
	char strRtspMediaURL[1024] = { 0 };
	char strSnapShotURL[1024] = { 0 };
	int iRet = 0;
	S_Stream_CTX*      pStreamCtx = NULL;

	do 
	{
		if (iStreamIdx == 0 || iStreamIdx == 1)
		{
			LOGFMTI("open channel:%d, stream:%d\n", iChannelIndex, (int)iStreamIdx);
			GetCurMediaURLS(pSdkIns, iChannelIndex, strRtspMediaURL, strSnapShotURL, iStreamIdx);
			pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIdx];
			if (pStreamCtx != NULL)
			{
				if (pStreamCtx->sMediaConvIns.iRunFlag == 1)
				{
					LOGFMTI("stream index:%d\n already open", (int)iStreamIdx);
				}
				else
				{
					//Reset pStreamCtx Context
					ResetCTXForStreamCTX(pStreamCtx);
					StartMediaPush(pSdkIns, iChannelIndex, strRtspMediaURL, "", iStreamIdx);
				}
			}
			else
			{
				iRet = AddStreamCtxSDKIns(pSdkIns, iChannelIndex, iStreamIdx);
				if (iRet != 0)
				{
					LOGFMTI("create channel %, stream index:%d context error, ", iChannelIndex, iStreamIdx);
				}
				else
				{
					pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIdx];
					StartMediaPush(pSdkIns, iChannelIndex, strRtspMediaURL, "", iStreamIdx);
				}
			}
		}
		else
		{
			LOGFMTE("Invalid stream index:%d\n", (int)iStreamIdx);
		}

	} while (0);
}

static void  HandleUpdateChannelInfo(S_SDK_Ins*  pSdkIns, int iChannelIndex)
{
	int iIndex = 0;
	int iStreamCount = 0;
	int iRet = 0;
	S_Channel_Map_Info*  pChannlMapInfoNew = NULL;
	S_Channel_Map_Info*  pChannlMapInfoCur = NULL;
	char   strChannelURLInner[1024] = { 0 };

	do 
	{
		printf("in update, channel index:%d\n", iChannelIndex);
		pChannlMapInfoCur = pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex];
		if (pChannlMapInfoCur == NULL)
		{
			break;
		}
		
		pChannlMapInfoNew = (S_Channel_Map_Info*)malloc(sizeof(S_Channel_Map_Info));
		memset(pChannlMapInfoNew, 0, sizeof(S_Channel_Map_Info));
		iRet = GetChannelInfoFromBZSetting(strChannelURLInner, pChannlMapInfoNew, iChannelIndex);
		if (iRet == 0)
		{
			if (strcmp(pChannlMapInfoNew->sOnvifInfo.strIP, pChannlMapInfoCur->sOnvifInfo.strIP) == 0 &&
				strcmp(pChannlMapInfoNew->sOnvifInfo.strUser, pChannlMapInfoCur->sOnvifInfo.strUser) == 0 &&
				strcmp(pChannlMapInfoNew->sOnvifInfo.strPwd, pChannlMapInfoCur->sOnvifInfo.strPwd) == 0)
			{
				//Just Change the Desc
				memset(pChannlMapInfoCur->strNodeDesc, 0, sizeof(pChannlMapInfoCur->strNodeDesc));
				strcpy(pChannlMapInfoCur->strNodeDesc, pChannlMapInfoNew->strNodeDesc);
			}
			else
			{
				//IP, UID, PWD change
				iStreamCount = pChannlMapInfoCur->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount;
				for (iIndex = 0; iIndex < iStreamCount && iIndex < MAX_SMARTBOX_STREAM_COUNT; iIndex++)
				{
					StopMediaPush(pSdkIns, iChannelIndex, iIndex);
				}

				memset(pChannlMapInfoCur, 0, sizeof(pChannlMapInfoCur));
				memcpy(pChannlMapInfoCur, pChannlMapInfoNew, sizeof(S_Channel_Map_Info));
			}
		}
	} while (0);


	printf("update Channel Detail Info, channel index:%d\n", iChannelIndex);
}



static int AddChannelToSmartBox(S_SDK_Ins*  pSdkIns, int iChannelIndex)
{
	int iRet = 0;
	Dev_Channel_Info_t* pChannel = NULL;
	S_Channel_Map_Info*  pChannlMapInfo = NULL;
	Dev_Stream_Info_t* pStream = NULL;
	int iStreamCount = 0;
	int iIdx = 0;

	do
	{
		pChannel = &((pSdkIns->sDevInfo).channel_list[iChannelIndex]);
		pChannlMapInfo = pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex];
		pChannel->channel_index = iChannelIndex + 1;
		if (pChannlMapInfo != NULL)
		{
			if (pChannlMapInfo->iChannelNodeType == CM_NODE_ONVIF)
			{
				if (pChannlMapInfo->ulChannelNodeState & ((1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS) | (1 << E_NODE_STATE_CHANNEL_ENABLE)))
				{
					pChannel->channel_status = EN_CH_STS_ONLINE;
					pChannel->stream_num = pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount;
					pChannel->stream_list = (Dev_Stream_Info_t*)malloc(sizeof(Dev_Stream_Info_t)*pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount);
					iStreamCount = pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount;
					for (iIdx = 0; iIdx < iStreamCount; iIdx++)
					{
						pStream = &pChannel->stream_list[iIdx];
						memset(pStream, 0, sizeof(Dev_Stream_Info_t));
						pStream->stream_id = iIdx;
						pStream->video_height = pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIdx].iHeight;
						pStream->video_width = pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iIdx].iWidth;
						pStream->video_codec.codec_fmt = EN_VIDEO_FMT_H264;
					}
				}
				else
				{
					pChannel->channel_status = EN_CH_STS_OFFLINE;
				}
			}

			if (pChannlMapInfo->iChannelNodeType == CM_NODE_STREAM_URL)
			{
				if (pChannlMapInfo->ulChannelNodeState & (1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS))
				{
					pChannel->channel_status = EN_CH_STS_ONLINE;
					pChannel->stream_num = 1;
					pChannel->stream_list = (Dev_Stream_Info_t*)malloc(sizeof(Dev_Stream_Info_t));
					pStream = &(pChannel->stream_list[0]);
					memset(pStream, 0, sizeof(Dev_Stream_Info_t));
					pStream->video_height = pChannlMapInfo->sStreamInfo.iHeight;
					pStream->video_width = pChannlMapInfo->sStreamInfo.iWidth;

					if (pChannlMapInfo->sStreamInfo.iVideoCodec == (int)AV_CODEC_ID_H264)
					{
						pStream->video_codec.codec_fmt = EN_VIDEO_FMT_H264;
					}

					if (pChannlMapInfo->sStreamInfo.iVideoCodec == (int)AV_CODEC_ID_H265)
					{
						pStream->video_codec.codec_fmt = EN_VIDEO_FMT_H265;
					}

					if (pChannlMapInfo->sStreamInfo.iAudioCodec == (int)AV_CODEC_ID_AAC)
					{
						pChannel->adudo_codec.codec_fmt = EN_AUDIO_FMT_AAC;
					}

					if (pChannlMapInfo->sStreamInfo.iAudioCodec == (int)AV_CODEC_ID_PCM_ALAW)
					{
						pChannel->adudo_codec.codec_fmt = EN_AUDIO_FMT_G711_A;
					}

					if (pChannlMapInfo->sStreamInfo.iAudioCodec == (int)AV_CODEC_ID_PCM_MULAW)
					{
						pChannel->adudo_codec.codec_fmt = EN_AUDIO_FMT_G711_U;
					}

					pChannel->adudo_codec.bitwidth = (pChannlMapInfo->sStreamInfo.iAudioBitWidth == 8)?0:1;
					pChannel->adudo_codec.channel = (pChannlMapInfo->sStreamInfo.iAudioChannel == 1) ? (EN_AUDIO_CH_MONO) : (EN_AUDIO_CH_STEREO);
					pChannel->adudo_codec.sample = GetSampleRateValue(pChannlMapInfo->sStreamInfo.iAudioSampleRate);
					memcpy(pChannel->adudo_codec.sepc_data, pChannlMapInfo->sStreamInfo.uAudioExtra, pChannlMapInfo->sStreamInfo.iAudioExtraSize);
					pChannel->adudo_codec.sepc_size = pChannlMapInfo->sStreamInfo.iAudioExtraSize;
				}
			}
			else
			{
				pChannel->channel_status = EN_CH_STS_OFFLINE;
			}
		}
	} while (0);

	return iRet;
}


static void HandleAddChannel(S_SDK_Ins*  pSdkIns, int iChannelIndex)
{
	S_Channel_Map_Info*   pChannelInfo = NULL;
	int iRet = 0;
	int iStreamIdx = 0;
	char   strChannelURLInner[1024] = { 0 };

	do
	{
		printf("Add Channel Index:%d\n", iChannelIndex);
		sprintf(strChannelURLInner, pSdkIns->strBZChannelInfoReqURL, pSdkIns->sDevInfo.dev_id, iChannelIndex + 1);
		pChannelInfo = (S_Channel_Map_Info*)malloc(sizeof(S_Channel_Map_Info));
		if (pChannelInfo == NULL)
		{
			LOGE("Lack of memory!");
			break;
		}

		memset(pChannelInfo, 0, sizeof(S_Channel_Map_Info));
		iRet = GetChannelInfoFromBZSetting(strChannelURLInner, pChannelInfo, iChannelIndex);
		if (iRet == 0)
		{
			iRet = AddChannelNodeToBox(pSdkIns->pSmartBoxInfo, pChannelInfo, iChannelIndex);
			if (iRet != 0)
			{
				LOGFMTE("Add channel %d error", iChannelIndex);
				break;
			}

			pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->iActiveProfileIndex = -1;
			pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex] = pChannelInfo;
			CheckChannelNodeState(pSdkIns->pSmartBoxInfo, iChannelIndex);

			AddChannelToSmartBox(pSdkIns, iChannelIndex);
			Dev_Sdk_Channel_Status_Report(iChannelIndex+1,  &(pSdkIns->sDevInfo.channel_list[iChannelIndex]));
			if (pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->ulChannelNodeState & (1 << E_NODE_STATE_LOCAL_ONLINE_FLAG_POS))
			{
				//Select the Stream idx first
				iStreamIdx = SelectChannelNodeProfileIndex(pSdkIns->pSmartBoxInfo, iChannelIndex, 1024, 1280, 720);
				if (iStreamIdx != -1)
				{
					//For Convenient Index, the server begin index from 1
					AddStreamCtxSDKIns(pSdkIns, iChannelIndex, iStreamIdx);
					if (iStreamIdx != -1)
					{
						LOGFMTI("the select rtsp url:%s\n", pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIdx].strMainStreamRtspURL);
						LOGFMTI("the select snapshot url:%s\n", pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIdx].strMainStreamSnapshotURL);
					}

					//Remove the default push video
					//if (pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIdx] != NULL)
					//{
					//	iRet = StartMediaPush(pSdkIns, iChannelIndex, NULL, NULL, iStreamIdx);
					//}
				}
			}
		}
		else
		{
			LOGFMTE("Get Channel Index:%d Channel Info From BZ Error", iChannelIndex);
		}
	} while (0);

	if (pChannelInfo != NULL)
	{
		free(pChannelInfo);
	}

	return;
}


static void HandleRemoveChannel(S_SDK_Ins*  pSdkIns, int iChannelIndex)
{
	S_Channel_Map_Info*   pChannelInfo = NULL;
	int iRet = 0;
	int iStreamCount = 0;
	int iIndex = 0;

	do 
	{
		iRet = DelChannelFromSmartBox(pSdkIns, iChannelIndex);
		LOGFMTI("Remove Channel %d, ret value:%d", iChannelIndex, iRet);
	} while (0);

	return;
}


static void HandleCloseStream(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIdx)
{
	char strRtspMediaURL[1024] = { 0 };
	char strSnapShotURL[1024] = { 0 };
	int iRet = 0;
	S_Stream_CTX*      pStreamCtx = NULL;

	do
	{
		if (iStreamIdx == 0 || iStreamIdx == 1)
		{
			LOGFMTI("close stream index:%d\n", (int)iStreamIdx);
			GetCurMediaURLS(pSdkIns, iChannelIndex, strRtspMediaURL, strSnapShotURL, iStreamIdx);
			pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIdx];
			if (pStreamCtx != NULL)
			{
				if (pStreamCtx->sMediaConvIns.iRunFlag == 1)
				{
					StopMediaPush(pSdkIns, iChannelIndex, iStreamIdx);
				}
				else
				{
					LOGFMTI("stream index:%d\n already closed", (int)iStreamIdx);
				}
			}
			else
			{
				LOGFMTI("stream index:%d\n already closed", (int)iStreamIdx);
			}
		}
		else
		{
			LOGFMTE("Invalid stream index:%d\n", (int)iStreamIdx);
		}

	} while (0);

	return;
}

static void HandleUpdateChannel(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iCmdArg)
{
	int iRet = 0;
	do 
	{
		switch (iCmdArg)
		{
			//add channel
			case 1:
			{
				HandleAddChannel(pSdkIns, iChannelIndex);
				break;
			}

			//updata channel
			case 2:
			{
				HandleUpdateChannelInfo(pSdkIns, iChannelIndex);
				break;
			}

			//remove channel
			case 3:
			{
				HandleRemoveChannel(pSdkIns, iChannelIndex);
				break;
			}


			default:
			{
				break;
			}
		}

	} while (0);

	return;
}




void SdkAppCallback(void* user_data, Dev_Cmd_Param_t *args)
{
	S_SDK_Ins*  pSdkIns = NULL;
	int iChannelIdx = 0;
	int iRet = 0;
	int iImageSize = 0;
	char  strRtspMediaURL[1024] = { 0 };
	char  strSnapShotURL[1024] = { 0 };
	int iChannelIdxInner = 0;

	do 
	{
		LOGI("Enter Callback!\n");
		if (!user_data || !args)
		{
			break;
		}

		pSdkIns = (S_SDK_Ins*)user_data;
		iChannelIdx = args->channel_index;

		//for convention, the index start from 1, so minus 1
		if (pSdkIns->pStreamCtxArray == NULL || iChannelIdx < 1 || iChannelIdx > pSdkIns->pSmartBoxInfo->iMaxChannelNodeCount)
		{
			return;
		}

		pthread_mutex_lock(&(pSdkIns->sMutex));
		if (pSdkIns->iCmdCount < MAX_SMARTBOX_CACHE_MSG_COUNT)
		{
			pSdkIns->iCmdCount++;
			memcpy(pSdkIns->pDevCmd[pSdkIns->iCurCmdWritedx], args, sizeof(Dev_Cmd_Param_t));
			printf("in write cmd index %d, channel index:%d\n", pSdkIns->iCurCmdWritedx, pSdkIns->pDevCmd[pSdkIns->iCurCmdWritedx]->channel_index);
			pSdkIns->iCurCmdWritedx = (pSdkIns->iCurCmdWritedx + 1) % MAX_SMARTBOX_CACHE_MSG_COUNT;
		}
		pthread_mutex_unlock(&(pSdkIns->sMutex));
	} while (0);


	LOGI("Quit Callback!\n");
	return ;
}


int InitSDK(S_SDK_Ins*  pSdkIns, S_SmartBox_Info*  pSmartBoxInfo, char*  pDevCfg, char* pDeviceId, char* pEntryServerIp, int iEntryServerPort, int iMaxChannelNodeCount)
{
	int iIndex = 0;
	int iIndexInner = 0;
	int iRet = 0;
	Entry_Serv_t entryServ;

#if (OS_LINUX == 1)
	signal(SIGPIPE, SIG_IGN);
#endif

	memset(pSdkIns, 0x0, sizeof(S_SDK_Ins));
	pSdkIns->pSmartBoxInfo = pSmartBoxInfo;
	strcpy(pSdkIns->sDevInfo.dev_id, pDeviceId);
	pSdkIns->sDevInfo.dev_type = 4;
	pSdkIns->sDevInfo.channel_num = iMaxChannelNodeCount;
	pSdkIns->sDevInfo.oem_info.OEMID = 100827;
	strcpy(pSdkIns->sDevInfo.oem_info.OEM_name, "ZZ");
	strcpy(pSdkIns->sDevInfo.oem_info.Model, "PCCamera_01");
	strcpy(pSdkIns->sDevInfo.oem_info.Factory, "QN");

	pSdkIns->sDevInfo.attr.has_hard_disk = 0;
	pSdkIns->sDevInfo.attr.has_microphone = 0;
	pSdkIns->sDevInfo.attr.can_recv_audio = 0;
	pSdkIns->pSmartBoxInfo->iMaxChannelNodeCount = iMaxChannelNodeCount;
	SmartBoxChannelInfoConv(pSdkIns, pSmartBoxInfo);


	strcpy(entryServ.ip, pEntryServerIp);
	entryServ.port = iEntryServerPort;

	//Set the log config
	strcpy(pSdkIns->sDevInfo.log_path, "./");
	pSdkIns->sDevInfo.log_level = 4;
	pSdkIns->sDevInfo.log_max_size = 20000;
	pSdkIns->sDevInfo.log_backup_flag = 1;

	if (Dev_Sdk_Init(&entryServ, &(pSdkIns->sDevInfo)) < 0)
	{
		return -1;
	}

	Dev_Sdk_Set_CB(SdkAppCallback, pSdkIns);

	//Memory

	for (iIndex = 0; iIndex < MAX_SMARTBOX_DEVICE_COUNT; iIndex++)
	{
		for (iIndexInner = 0; iIndexInner < MAX_SMARTBOX_STREAM_COUNT; iIndexInner++)
		{
			pSdkIns->pStreamCtxArray[iIndex][iIndexInner] = NULL;
		}
	}

	for (iIndex = 0; iIndex < MAX_SMARTBOX_CACHE_MSG_COUNT; iIndex++)
	{
		pSdkIns->pDevCmd[iIndex] = (Dev_Cmd_Param_t*)malloc(sizeof(Dev_Cmd_Param_t));
		if (pSdkIns->pDevCmd[iIndex] == NULL)
		{
			LOGE("Lack of memory");
			return ERR_LACK_MEMORY;
		}
		else
		{
			memset(pSdkIns->pDevCmd[iIndex], 0, sizeof(Dev_Cmd_Param_t));
		}
	}


	pSdkIns->iCurStreamCount = 0;
	pSdkIns->iCurCmdReadIdx = 0;
	pSdkIns->iCurCmdWritedx = 0;
	pSdkIns->iCmdCount = 0;

	iRet = pthread_mutex_init(&(pSdkIns->sMutex), NULL);
	if (iRet != 0)
	{
		LOGFMTE("init pthread mutex error!error code%d\n", iRet);
	}

	return iRet;
}

char* GetDeviceId(S_SDK_Ins* pSdkIns)
{
	return Dev_Sdk_Get_Device_ID();
}

int SdkStreamInfoReport(S_SDK_Ins*  pSdkIns, S_Frame*   pFrame, int iChannelIndex, int iStreamIndex)
{
	int iRet = 0;
	Dev_Stream_Frame_t  frame_envet;
	S_Stream_CTX*      pStreamCtx = NULL;
	time_t    tTime;
	do 
	{
		if (pSdkIns == NULL || pFrame == NULL)
		{
			break;
		}

		if (pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIndex] == NULL)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}
		pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIndex];

		memset(&frame_envet, 0x0, sizeof(Dev_Stream_Frame_t));
		frame_envet.frame_type = (unsigned short)pFrame->iFrameType;
		frame_envet.stream_id = iStreamIndex;
		frame_envet.frame_id = pStreamCtx->ulTotalFrmNum++;


		//For convention, channel index start from 1, so add 1
		frame_envet.channel_index = iChannelIndex+1;
		switch (pFrame->iMediaType)
		{
			case MEDIA_VIDEO_TYPE:
			{
				frame_envet.frame_av_id = pStreamCtx->ulVideoFrmNum++;
				break;
			}
			case MEDIA_AUDIO_TYPE:
			{
				frame_envet.frame_av_id = pStreamCtx->ulAudioFrmNum++;
				frame_envet.frame_type = EN_FRM_TYPE_AU;
				break;
			}
		}

		time(&(tTime));
		if (tTime > (pStreamCtx->lLastActiveTime + MIN_LOG_OUTPUT_INTERVAL))
		{
			pStreamCtx->lLastActiveTime = (long)tTime;
			LOGFMTI("device_id %s , channel_idx %d, stream_idx %d send data to SDK", pSdkIns->sDevInfo.dev_id, iChannelIndex, iStreamIndex);
		}
		frame_envet.frame_ts = pFrame->ulTimeStamp;
		frame_envet.frame_size = pFrame->iFrameSize;
		frame_envet.frame_offset = 0;
		frame_envet.pdata = pFrame->pFrameData;

		pStreamCtx->iFrmBufNum = Dev_Sdk_Stream_Frame_Report(&frame_envet);
	} while (0);

	return iRet;
}

int UnInitSDK(S_SDK_Ins*  pSdkIns)
{
	int iIndex = 0;
	int iIndexInner = 0;
	S_Stream_CTX*      pStreamCtx = NULL;

	for (iIndex = 0; iIndex < MAX_SMARTBOX_DEVICE_COUNT; iIndex++)
	{
		for (iIndexInner = 0; iIndexInner < MAX_SMARTBOX_STREAM_COUNT; iIndexInner++)
		{
			if (pSdkIns != NULL && pSdkIns->pStreamCtxArray[iIndex][iIndexInner] != NULL)
			{
				if (pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pImageBuffer != NULL)
				{
					free(pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pImageBuffer);
					pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pImageBuffer = NULL;
				}

				if (pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pFrameBuffer != NULL)
				{
					free(pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pImageBuffer);
					pSdkIns->pStreamCtxArray[iIndex][iIndexInner]->pFrameBuffer = NULL;
				}

				free(pSdkIns->pStreamCtxArray[iIndex][iIndexInner]);
				pSdkIns->pStreamCtxArray[iIndex][iIndexInner] = NULL;
			}
		}
	}

	for (iIndex = 0; iIndex < MAX_SMARTBOX_CACHE_MSG_COUNT; iIndex++)
	{
		if (pSdkIns != NULL && pSdkIns->pDevCmd[iIndex] != NULL)
		{
			free(pSdkIns->pDevCmd[iIndex]);
			pSdkIns->pDevCmd[iIndex] = NULL;
		}
	}

	pthread_mutex_destroy(&(pSdkIns->sMutex));

	return 0;
}

int GetCurMediaURLS(S_SDK_Ins*  pSdkIns, int iChannelIndex, char* pRtspMediaURL, char*  pSnapShotURL, int iStreamIndex)
{
	int iRet = 0;
	int iActiveIndex = 0;

	do 
	{
		if (pSdkIns == NULL || pRtspMediaURL == NULL || pSnapShotURL == NULL)
		{
			iRet = 1;
			break;
		}

		if (pSdkIns->pSmartBoxInfo->pChannelArray != NULL &&
			pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex] != NULL)

		{
			switch (pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->iChannelNodeType)
			{
				case CM_NODE_ONVIF:
				{
					if (iStreamIndex >= 0 &&
						iStreamIndex < pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount)
					{
						strcpy(pRtspMediaURL, pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIndex].strMainStreamRtspURL);
						strcpy(pSnapShotURL, pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sOnvifInfo.sDevOnvifInfo.sMediaProfiles[iStreamIndex].strMainStreamSnapshotURL);
					}
					break;
				}

				case CM_NODE_STREAM_URL:
				{
					strcpy(pRtspMediaURL, pSdkIns->pSmartBoxInfo->pChannelArray[iChannelIndex]->sStreamInfo.strStreamURL);
					break;
				}
			}
		}
		else
		{
			iRet = -1;
		}
	} while (0);

	return iRet;
}


int AddStreamCtxSDKIns(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIndex)
{
	int iRet = 0;
	S_Stream_CTX*  pStreamCtx = NULL;

	do 
	{
		pStreamCtx = (S_Stream_CTX*)malloc(sizeof(S_Stream_CTX));
		if (pStreamCtx == NULL)
		{
			iRet = -1;
			break;
		}
		memset(pStreamCtx, 0, sizeof(S_Stream_CTX));

		pStreamCtx->pFrameBuffer = (unsigned char*)malloc(1024 * 1024);
		pStreamCtx->iFrameBufMaxSize = 1024 * 1024;

		pStreamCtx->pImageBuffer = (unsigned char*)malloc(512 * 1024);
		pStreamCtx->iFrameBufMaxSize = 512 * 1024;

		pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIndex] = pStreamCtx;

		printf("channel index:%d, stream index:%d, malloc memory done!\n", iChannelIndex, iStreamIndex);
	} while (0);

	return iRet;
}


int SmartBoxChannelInfoConv(S_SDK_Ins*  pSdkIns, S_SmartBox_Info*  pSmartBoxInfo)
{
	int iRet = 0;
	int iChannelCount = 0;
	int iChannelMaxCount = 0;
	int iIndex = 0;
	int iIndex2 = 0;
	Dev_Channel_Info_t* pChannel = NULL;
	S_Channel_Map_Info*  pChannlMapInfo = NULL;
	Dev_Stream_Info_t* pStream = NULL;
	int iStreamCount = 0;

	do 
	{
		if (pSdkIns == NULL || pSmartBoxInfo == NULL)
		{
			iRet = -1;
			break;
		}

		iChannelMaxCount = pSmartBoxInfo->iMaxChannelNodeCount;
		pSdkIns->sDevInfo.channel_list = (Dev_Channel_Info_t*)malloc(sizeof(Dev_Channel_Info_t)*iChannelMaxCount);
		memset(pSdkIns->sDevInfo.channel_list, 0, sizeof(Dev_Channel_Info_t)*iChannelMaxCount);
		for (iIndex = 0; iIndex < iChannelMaxCount; iIndex++)
		{
			AddChannelToSmartBox(pSdkIns, iIndex);
		}
	} while (0);

	return iRet;
}

int TransactMediaFrameFromFFmpeg(void*  pFrameFromFFmpeg, void* pMediaConvCtx, void*  pSdkIns, int iChannelIndex, int iStreamIndex)
{
	int iRet = 0;
	AVPacket*   pPkg = (AVPacket*)(pFrameFromFFmpeg);
	S_SDK_Ins*  pSdk = (S_SDK_Ins*)pSdkIns;
	AVFormatContext *pIfmtCtx = NULL;
	S_Media_Conv* pConvCtx = (S_Media_Conv*)pMediaConvCtx;
	S_Stream_CTX*   pStreamCtx = NULL;

	S_Frame     sFrame;
	int iPreFrameSize = 0;
	int iKeyFrameFlag = 0;

	do 
	{
		if (pPkg == NULL || pSdk == NULL || pMediaConvCtx == NULL)
		{
			iRet = -1;
			break;
		}

		if (pSdk->pStreamCtxArray != NULL && pSdk->pStreamCtxArray[iChannelIndex] != NULL)
		{
			pStreamCtx = pSdk->pStreamCtxArray[iChannelIndex][iStreamIndex];
		}
		else
		{
			iRet = -1;
			break;
		}

		memset(&sFrame, 0, sizeof(S_Frame));
		sFrame.pFrameData = pStreamCtx->pFrameBuffer;
		
		pIfmtCtx = pConvCtx->pIfmtCtx;


		sFrame.ulTimeStamp = (unsigned int)((pPkg->pts * av_q2d(pIfmtCtx->streams[pPkg->stream_index]->time_base)) * 1000);
		iPreFrameSize = pPkg->size;
		if (pIfmtCtx->streams[pPkg->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && pIfmtCtx->streams[pPkg->stream_index]->codecpar->codec_id == AV_CODEC_ID_H264)
		{

			//avc mode to h264, 7 indicate h264
			if (pConvCtx->iNalLengthSize > 0)
			{
				qcAV_ConvertAVCNalFrame(sFrame.pFrameData, sFrame.iFrameSize, pPkg->data, pPkg->size, pConvCtx->iNalLengthSize, iKeyFrameFlag, 7);
				if (sFrame.iFrameSize != 0)
				{
					memcpy(sFrame.pFrameData, pPkg->data, pPkg->size);
				}
				else if (iPreFrameSize == pPkg->size)
				{
					memcpy(sFrame.pFrameData, pPkg->data, pPkg->size);
				}
				sFrame.iFrameSize = pPkg->size;
			}
			else
			{
				memcpy(sFrame.pFrameData, pPkg->data, pPkg->size);
				sFrame.iFrameSize = pPkg->size;
			}

			sFrame.iMediaType = MEDIA_VIDEO_TYPE;
			if ((pPkg->flags & AV_PKT_FLAG_KEY) != 0)
			{
				sFrame.iFrameType = FRM_VIDEO_I_TYPE;
			}
			else
			{
				sFrame.iFrameType = FRM_VIDEO_P_TYPE;
			}
			//
			//if (_gVideoDataDump[iChannelIndex] == NULL)
			//{
			//	char strVDataDump[128] = { 0 };
			//	char strVInfoDump[128] = { 0 };
			//	sprintf(strVDataDump, "ch_data_%d.h264", iChannelIndex);
			//	sprintf(strVInfoDump, "ch_info_%d.txt", iChannelIndex);

			//	_gVideoDataDump[iChannelIndex] = fopen(strVDataDump, "wb");
			//	_gVideoInfoDump[iChannelIndex] = fopen(strVInfoDump, "wb");
			//}

			//if (_gVideoDataDump[iChannelIndex] != NULL && _gVideoInfoDump[iChannelIndex] != NULL)
			//{
			//	char strFrameInfo[256] = { 0 };
			//	fwrite(sFrame.pFrameData, 1, sFrame.iFrameSize, _gVideoDataDump[iChannelIndex]);
			//	sprintf(strFrameInfo, "frame size:%d, frame time:%d, frame type:%d\n", sFrame.iFrameSize, (int)sFrame.ulTimeStamp, sFrame.iFrameType);
			//	fwrite(strFrameInfo, 1, strlen(strFrameInfo), _gVideoInfoDump[iChannelIndex]);
			//	fflush(_gVideoDataDump[iChannelIndex]);
			//	fflush(_gVideoInfoDump[iChannelIndex]);
			//}
			//
			
			SdkStreamInfoReport(pSdk, &sFrame, iChannelIndex, iStreamIndex);
		}

		if (pIfmtCtx->streams[pPkg->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && pIfmtCtx->streams[pPkg->stream_index]->codecpar->codec_id == AV_CODEC_ID_AAC)
		{
			memcpy(sFrame.pFrameData, pPkg->data, pPkg->size);
			sFrame.iFrameSize = pPkg->size;
			sFrame.iMediaType = MEDIA_AUDIO_TYPE;
			sFrame.iFrameType = 0;
			SdkStreamInfoReport(pSdk, &sFrame, iChannelIndex, iStreamIndex);
		}

	} while (0);
	return 0;
}

int StartMediaPush(S_SDK_Ins*  pSdkIns, int iChannelIndex, char* pOutuptURL, char* pOutputFmt, int iStreamIndex)
{
	int iRet = 0;
	S_Stream_CTX*  pStreamCtx = NULL;
	char strInputMediaURL[1024] = { 0 };
	char strSnapshotURL[1024] = { 0 };

	do 
	{
		if (pSdkIns != NULL && pSdkIns->pStreamCtxArray != NULL && pSdkIns->pStreamCtxArray[iChannelIndex] != NULL)
		{
			pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIndex];
		}
		else
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		iRet = GetCurMediaURLS(pSdkIns, iChannelIndex, strInputMediaURL, strSnapshotURL, iStreamIndex);
		if (iRet != 0)
		{
			iRet = ERR_INVALID_PARAMETER;
			LOGE("Can't get the Media URL!\n");
			break;
		}

		iRet = InitMediaConv(&(pStreamCtx->sMediaConvIns), strInputMediaURL, pOutuptURL, pOutputFmt, pSdkIns, iChannelIndex, iStreamIndex);
		if (iRet != 0)
		{
			LOGE("Init MediaConv Fail!\n");
			iRet = ERR_OPEN_URL_FAIL;
			break;
		}

		strcpy(pStreamCtx->sMediaConvIns.strInputURL, strInputMediaURL);
		if (pOutuptURL != NULL && strlen(pOutuptURL) != 0)
		{
			strcpy(pStreamCtx->sMediaConvIns.strOutputFmt, pOutuptURL);
			strcpy(pStreamCtx->sMediaConvIns.strOutputFmt, pOutputFmt);
		}

		iRet = pthread_create(&(pStreamCtx->hPthreadHandle), NULL, ThreadFuncForMediaConv, &(pStreamCtx->sMediaConvIns));
		if (iRet != 0)
		{
			LOGE("Start Media push Thread fail!\n");
			iRet = ERR_OPEN_URL_FAIL;
		}

	} while (0);

	return iRet;
}

int StopMediaPush(S_SDK_Ins*  pSdkIns, int iChannelIndex, int iStreamIndex)
{
	int iRet = 0;
	S_Stream_CTX*  pStreamCtx = NULL;
	S_Media_Conv*       pMediaConvIns = NULL;
	void*               pThreadRet = NULL;

	do
	{
		if (pSdkIns != NULL && pSdkIns->pStreamCtxArray != NULL && pSdkIns->pStreamCtxArray[iChannelIndex] != NULL)
		{
			pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIndex][iStreamIndex];
		}
		else
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		if (pStreamCtx != NULL)
		{
			pMediaConvIns = &(pStreamCtx->sMediaConvIns);
			if (pMediaConvIns->iRunFlag == 1)
			{
				pMediaConvIns->iRunFlag = 0;
			}

			pthread_join(pStreamCtx->hPthreadHandle, &pThreadRet);
			UnInitMediaConv(pMediaConvIns);
		}
	} while (0);

	return iRet;
}

int GetCmdItem(S_SDK_Ins*  pSdkIns, Dev_Cmd_Param_t *args)
{
	int iRet = -1;

	do 
	{
		if (pSdkIns == NULL)
		{
			break;
		}

		pthread_mutex_lock(&(pSdkIns->sMutex));
		if (pSdkIns->iCmdCount > 0)
		{
			printf("cur Cmd index:%d\n", pSdkIns->iCurCmdReadIdx);
			memcpy(args, pSdkIns->pDevCmd[pSdkIns->iCurCmdReadIdx], sizeof(Dev_Cmd_Param_t));
			printf("out cur channel index:%d\n", args->channel_index);
			pSdkIns->iCmdCount--;
			pSdkIns->iCurCmdReadIdx = (pSdkIns->iCurCmdReadIdx + 1) % MAX_SMARTBOX_CACHE_MSG_COUNT;
			iRet = 0;
		}
		pthread_mutex_unlock(&(pSdkIns->sMutex));

	} while (0);

	return iRet;
}

int HandleCmd(S_SDK_Ins*  pSdkIns, Dev_Cmd_Param_t *args)
{
	char  strRtspMediaURL[1024] = { 0 };
	char  strSnapShotURL[1024] = { 0 };
	int   iStreamIdx = 0;
	int   iImageSize = 0;
	int   iRet = 0;
	unsigned short   usCmdValue = 0;

	int  iChannelIdxInner = 0;
	S_Stream_CTX*      pStreamCtx = NULL;

	iChannelIdxInner = args->channel_index - 1;
	if (args->cmd_type == EN_CMD_LIVE_OPEN)
	{
		iStreamIdx = (int)args->cmd_args[0];
		printf("Handle Open\n");
		HandleOpenStream(pSdkIns, iChannelIdxInner, iStreamIdx);
	}
	else if (args->cmd_type == EN_CMD_LIVE_CLOSE)
	{
		iStreamIdx = (int)args->cmd_args[0];
		printf("Handle Close\n");
		HandleCloseStream(pSdkIns, iChannelIdxInner, iStreamIdx);
	}
	else if (args->cmd_type == EN_CMD_MGR_UPDATE)
	{
		usCmdValue = (int)args->cmd_args[0];
		usCmdValue |= ((args->cmd_args[1]) << 8);
		HandleUpdateChannel(pSdkIns, iChannelIdxInner, (int)usCmdValue);
	}
	else if (args->cmd_type == EN_CMD_SNAP)
	{
		pStreamCtx = pSdkIns->pStreamCtxArray[iChannelIdxInner][0];

		do
		{
			GetCurMediaURLS(pSdkIns, iChannelIdxInner, strRtspMediaURL, strSnapShotURL, 0);
			if (strlen(strSnapShotURL) != 0)
			{
				iRet = GetCameraSnapShot(strSnapShotURL, &(pStreamCtx->pImageBuffer), &(pStreamCtx->iImageBufMaxSize), &iImageSize);
			}
		} while (0);

		if (iImageSize > 0)
		{
			LOGFMTE("Enter upload iamge, image size:%d\n", iImageSize);
			Dev_Sdk_Snap_Picture_Report(args->channel_index, EN_PIC_FMT_JPEG, (sdk_uint8*)pStreamCtx->pImageBuffer, iImageSize);
		}
	}

	return iRet;
}

int DelChannelFromSmartBox(S_SDK_Ins*  pSdkIns, int iIndex)
{
	int iStreamIdx = 0;
	int iRet = 0;
	Dev_Channel_Info_t* pChannel = NULL;
	S_Channel_Map_Info*  pChannlMapInfo = NULL;
	int iStreamCount = 0;

	do 
	{
		pChannel = &(pSdkIns->sDevInfo.channel_list[iIndex]);
		pChannlMapInfo = pSdkIns->pSmartBoxInfo->pChannelArray[iIndex];
		if (pChannlMapInfo == NULL || pChannel == NULL)
		{
			break;
		}

		iStreamCount = pChannlMapInfo->sOnvifInfo.sDevOnvifInfo.iMediaProfileCount;
		for (iStreamIdx = 0; iStreamIdx < iStreamCount && iStreamIdx < MAX_SMARTBOX_STREAM_COUNT; iStreamIdx++)
		{
			StopMediaPush(pSdkIns, iIndex, iStreamIdx);
			if (pSdkIns->pStreamCtxArray[iIndex][iStreamIdx] != NULL)
			{
				memset(pSdkIns->pStreamCtxArray[iIndex][iStreamIdx], 0, sizeof(S_Stream_CTX));
			}
		}

		pChannlMapInfo->ulChannelNodeState = 0;
		pChannel->channel_status = EN_CH_STS_OFFLINE;
		Dev_Sdk_Channel_Status_Report(iIndex + 1, pChannel);
	} while (0);

	return iRet;
}
