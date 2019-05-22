#include "MediaSimulator.h"
#include "qiniu_dev_net_porting.h"
#include "UAVParser.h"



FILE*  gpDumpH264 = NULL;
void InitMediaSimulator(S_Media_Simulator*  pMedaiSimu)
{
	do 
	{
		if (pMedaiSimu == NULL)
		{
			break;
		}

		memset(pMedaiSimu, 0, sizeof(S_Media_Simulator)); 
		pMedaiSimu->iAudioIndex = -1;
		pMedaiSimu->iVideoIndex = -1;
	} while (0);
}

void UnInitMediaSimulator(S_Media_Simulator*  pMedaiSimu)
{

}

int  OpenMedia(S_Media_Simulator*  pMedaiSimu, char* pMediaURL)
{
	int iRet = 0;
	AVDictionary*     pOptions = NULL;
	AVStream*         pAVStream = NULL;
	long long iTimeOutForLiveStream = 0;
	char    strTime[64] = { 0 };
	char    strErrorInfo[1024] = { 0 };
	unsigned char    aSync[4] = { 0, 0, 0, 1 };
	int iErr = 0;
	int iIndex = 0;
	int iParseHeader = 0;

	do
	{
		if (pMedaiSimu == NULL)
		{
			break;
		}

		if (strlen(pMediaURL) > 7 && memcmp(pMediaURL, "rtsp://", 7) == 0)
		{
			av_dict_set(&pOptions, "rtsp_flags", "prefer_tcp", 0);
			if (iTimeOutForLiveStream <= 0)
			{
				iTimeOutForLiveStream = 5;
			}

			sprintf(strTime, "%lld", iTimeOutForLiveStream * 1000000);
			av_dict_set(&pOptions, "stimeout", strTime, 0);
			av_dict_set(&pOptions, "fflags", "nobuffer", 0);
		}


		if ((iRet = avformat_open_input(&(pMedaiSimu->pIfmt_ctx), pMediaURL, 0, &pOptions)) < 0)
		{
			memset(strErrorInfo, 0, 1024);
			av_strerror(iRet, strErrorInfo, 1024);
			printf("Could not open input file: %s , error info:%s\n", pMediaURL, strErrorInfo);
			iRet = -1;
			iErr = 1;
			break;
		}

		if ((iRet = avformat_find_stream_info(pMedaiSimu->pIfmt_ctx, 0)) < 0)
		{
			memset(strErrorInfo, 0, 1024);
			av_strerror(iRet, strErrorInfo, 1024);
			printf("Failed to retrieve input stream information for %s , error info:%s\n", pMediaURL, strErrorInfo);
			iErr = 1;
			iRet = ERR_CANNOT_FIND_VIDEO;
			break;
		}

		for (iIndex = 0; iIndex < pMedaiSimu->pIfmt_ctx->nb_streams; iIndex++)
		{
			pAVStream = pMedaiSimu->pIfmt_ctx->streams[iIndex];
			switch (pAVStream->codecpar->codec_type)
			{
				case AVMEDIA_TYPE_VIDEO:
				{
					pMedaiSimu->iVideoIndex = iIndex;
					pMedaiSimu->pVideoStream = pAVStream;
					pMedaiSimu->eVideoCodecID = pAVStream->codecpar->codec_id;
					pMedaiSimu->iVideoWidth = pAVStream->codecpar->width;
					pMedaiSimu->iVideoHeight = pAVStream->codecpar->height;

					if (pAVStream->codecpar->codec_id == AV_CODEC_ID_H264)
					{
						if (pAVStream->codecpar->extradata != NULL && pAVStream->codecpar->extradata_size > 4 && memcmp(pAVStream->codecpar->extradata, aSync, 4) != 0)
						{
							iParseHeader = qcAV_ConvertAVCNalHead(pMedaiSimu->aVideoHeader, pMedaiSimu->iVideoHeaderSize, pAVStream->codecpar->extradata, pAVStream->codecpar->extradata_size, pMedaiSimu->iH264NalSizeLen);
						}
					}
					
					break;
				}

				case AVMEDIA_TYPE_AUDIO:
				{
					pMedaiSimu->iAudioIndex = iIndex;
					pMedaiSimu->pAudioStream = pAVStream;
					pMedaiSimu->eAudioCodecID = pAVStream->codecpar->codec_id;
					pMedaiSimu->iAudioChannels = pAVStream->codecpar->channels;
					pMedaiSimu->iAudioSampleRate = pAVStream->codecpar->sample_rate;
					break;
				}
			}
		}

	} while (0);

	if (iErr != 0)
	{
		iRet = ERR_OPEN_MEDIA_FAIL;
	}

	return iRet;
}

int  ReadFrame(S_Media_Simulator*  pMediaSimu, void*  pOutput)
{
	int iRet = 0;
	int iKeyFrameFlag = 0;
	AVPacket pkt;
	int iPreFrameSize = 0;
	S_Frame*   pFrameOutput = (S_Frame*)pOutput;
	do 
	{
		if (pMediaSimu == NULL || pFrameOutput == NULL)
		{
			break;
		}

		av_init_packet(&pkt);
		iRet = av_read_frame(pMediaSimu->pIfmt_ctx, &pkt);
		while (iRet == 0 && pkt.pts == AV_NOPTS_VALUE)
		{
			av_free_packet(&pkt);
			av_init_packet(&pkt);
			iRet = av_read_frame(pMediaSimu->pIfmt_ctx, &pkt);
		}

		if (iRet != 0)
		{
			printf("Read Frame Faild!\n");
			break;
		}

		pFrameOutput->ulTimeStamp = (unsigned int)((pkt.pts * av_q2d(pMediaSimu->pIfmt_ctx->streams[pkt.stream_index]->time_base)) * 1000);
		iPreFrameSize = pkt.size;
		if (pkt.stream_index == pMediaSimu->iVideoIndex )
		{
			if (pMediaSimu->pVideoStream->codecpar->codec_id == AV_CODEC_ID_H264 && pMediaSimu->iH264NalSizeLen > 0)
			{
				qcAV_ConvertAVCNalFrame(pFrameOutput->pFrameData, pFrameOutput->iFrameSize, pkt.data, pkt.size, pMediaSimu->iH264NalSizeLen, iKeyFrameFlag, 7);
			}

			if (pFrameOutput->iFrameSize != 0)
			{
				memcpy(pFrameOutput->pFrameData, pkt.data, pkt.size);
			}
			else if (iPreFrameSize == pkt.size)
			{
				memcpy(pFrameOutput->pFrameData, pkt.data, pkt.size);
			}

			printf("frame info, frame size:%d, frame time:%d\n", pFrameOutput->iFrameSize, pFrameOutput->ulTimeStamp);
			pFrameOutput->iFrameSize = pkt.size;
			pFrameOutput->iMediaType = FRM_MEDIA_VIDEO_TYPE;
			if ((pkt.flags & AV_PKT_FLAG_KEY) != 0)
			{
				pFrameOutput->iFrameType = CH_I_FRM;
			}
			else
			{
				pFrameOutput->iFrameType = CH_P_FRM;
			}
		}

		if (pkt.stream_index == pMediaSimu->iAudioIndex)
		{
			pFrameOutput->iMediaType = FRM_MEDIA_AUDIO_TYPE;
			pFrameOutput->iFrameType = CH_AUDIO_FRM;
		}

		DumpFrame(pFrameOutput);
		av_free_packet(&pkt);
	} while (0);

	return iRet;
}

int  CloseMedia(S_Media_Simulator*  pMedaiSimu)
{
	int iRet = 0;
	return iRet;
}

int  DumpFrame(void*  pOutput)
{
	S_Frame*   pFrameOutput = (S_Frame*)pOutput;

	if (gpDumpH264 == NULL)
	{
		gpDumpH264 = fopen("video.h264", "wb");
	}

	if (gpDumpH264 != NULL)
	{
		fwrite(pFrameOutput->pFrameData, 1, pFrameOutput->iFrameSize, gpDumpH264);
		fflush(gpDumpH264);
	}

	return 0;
}