#include "MediaConv.h"
#include "qiniu_dev_net_porting.h"
#include "UAVParser.h"
#include "log4z.h"

static int SetVideoCodec(AVFormatContext* pAvFmtCtx, int iVideoCodec, unsigned char*  pVideoHeaderData, int iVideoHeaderSize, int iWidth, int iHeight, AVStream** ppAVStream, int* pStreamCount, int* piVideoIndex)
{
	int iRet = 0;
	AVCodecParameters	sCodecCtx;
	AVStream *pOutStream = NULL;

	do
	{
		pOutStream = avformat_new_stream(pAvFmtCtx, NULL);
		if (!pOutStream)
		{
			LOGE("Failed allocating output stream\n");
			iRet = ERR_CREATE_VIDEO_STREAM_ERROR;
			break;
		}

		memset(&sCodecCtx, 0, sizeof(AVCodecParameters));
		*piVideoIndex = *pStreamCount;
		(*pStreamCount)++;

		sCodecCtx.codec_type = AVMEDIA_TYPE_VIDEO;
		sCodecCtx.height = iHeight;
		sCodecCtx.width = iWidth;
		sCodecCtx.codec_id = (AVCodecID)iVideoCodec;
		sCodecCtx.extradata = pVideoHeaderData;
		sCodecCtx.extradata_size = iVideoHeaderSize;
		iRet = avcodec_parameters_copy(pOutStream->codecpar, &(sCodecCtx));

		if (iRet < 0)
		{
			LOGE("Failed to copy context from input to output stream codec context");
			break;
		}

		pOutStream->codecpar->codec_tag = 0;
		if (pAvFmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
		{
			//out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
		}

		*ppAVStream = pOutStream;

	} while (false);

	return iRet;
}

static int SetAudioCodec(AVFormatContext* pAvFmtCtx, int iAudioCodec, unsigned char*  pAudioHeaderData, int iAudioHeaderSize, int iChannels, int iSampleRate, AVStream** ppAVStream, int* pStreamCount, int* piAudioIndex)
{
	int iRet = 0;
	AVCodecParameters	sCodecCtx;
	AVStream *pOutStream = NULL;

	do
	{
		pOutStream = avformat_new_stream(pAvFmtCtx, NULL);
		if (!pOutStream)
		{
			LOGE("Failed allocating output stream\n");
			iRet = ERR_CREATE_VIDEO_STREAM_ERROR;
			break;
		}

		memset(&sCodecCtx, 0, sizeof(AVCodecParameters));
		*piAudioIndex = *pStreamCount;
		(*pStreamCount)++;

		sCodecCtx.codec_type = AVMEDIA_TYPE_AUDIO;
		sCodecCtx.channels = iChannels;
		sCodecCtx.sample_rate = iSampleRate;
		sCodecCtx.codec_id = (AVCodecID)iAudioCodec;
		sCodecCtx.extradata = pAudioHeaderData;
		sCodecCtx.extradata_size = iAudioHeaderSize;
		iRet = avcodec_parameters_copy(pOutStream->codecpar, &(sCodecCtx));

		if (iRet < 0)
		{
			LOGE("Failed to copy context from input to output stream codec context");
			break;
		}

		pOutStream->codecpar->codec_tag = 0;
		if (pAvFmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
		{
			//out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
		}

		*ppAVStream = pOutStream;
	} while (false);

	return iRet;
}


static int PrepareOuptput(AVFormatContext* pAvFmtCtx, char*  pOutputPath)
{
	int iRet = 0;

	do
	{
		av_dump_format(pAvFmtCtx, 0, pOutputPath, 1);
		if (!(pAvFmtCtx->oformat->flags & AVFMT_NOFILE))
		{
			iRet = avio_open(&pAvFmtCtx->pb, pOutputPath, AVIO_FLAG_WRITE);
			if (iRet < 0)
			{
				LOGFMTE("Could not open output file '%s'\n", pOutputPath);
				break;
			}
		}

		iRet = avformat_write_header(pAvFmtCtx, NULL);
		if (iRet < 0)
		{
			LOGE("Error occurred when opening output file\n");
			break;
		}
	} while (false);

	//Send the FirstIFrameCome Flag to false; 
	return iRet;
}

int InitMediaConv(S_Media_Conv*   pMediaConv, char*  pInputURL, char* pOutputURL, char* pOutputFmt, void*  pPrivate, int iIndexForPrivate, int iIndexForPrivateSub)
{
	int iRet = 0;
	int iErr = 0;
	AVDictionary*     pOptions = NULL;
	int iTimeOutForLiveStream = 0;
	char   strTime[64] = { 0 };
	char   strErrInfo[1024] = { 0 };
	AVFormatContext *pIfmtCtx = NULL;
	AVFormatContext *pOfmtCtx = NULL;
	AVStream *pInputStream = NULL;
	AVStream *pVideoStream = NULL;
	AVStream *pAudioStream = NULL;
	AVStream *pOutputStream = NULL;
	int       iInputVideoIndex = -1;
	int       iInputAudioIndex = -1;
	int       iOutputStreamCount = 0;
	int       iOutputVideoIndex = -1;
	int       iOutputAudioIndex = -1;
	unsigned char   aSync[4] = { 0, 0, 0, 1 };
	unsigned char   aVideoHeader[256] = { 0 };
	int             iVideoHeaderSize = 0;
	int             iNalLengthSize = 0;
	int             iParsed = 0;
	int iIndex = 0;

	do 
	{

		if (pMediaConv == NULL || pInputURL == NULL )
		{
			iRet = ERR_INVALID_PARAMETERS;
			break;
		}
		else
		{
			if (pPrivate == NULL)
			{
				if (pOutputURL == NULL || pOutputFmt == NULL)
				{
					iRet = ERR_INVALID_PARAMETERS;
					break;
				}
			}
		}

		memset(pMediaConv, 0, sizeof(S_Media_Conv));
		if (strlen(pInputURL) > 7 && memcmp(pInputURL, "rtsp://", 7) == 0)
		{
			av_dict_set(&pOptions, "rtsp_flags", "prefer_tcp", 0);

			if (iTimeOutForLiveStream <= 0)
			{
				iTimeOutForLiveStream = 5;
			}
			sprintf(strTime, "%lld", (long long)iTimeOutForLiveStream * 1000000);
			av_dict_set(&pOptions, "stimeout", strTime, 0);
		}

		av_dict_set(&pOptions, "fflags", "nobuffer", 0);


		//Open Input URL
		if ((iRet = avformat_open_input(&pIfmtCtx, pInputURL, 0, &pOptions)) < 0)
		{
			memset(strErrInfo, 0, 1024);
			av_strerror(iRet, strErrInfo, 1024);
			LOGFMTE("Could not open input file: %s , error info:%s\n", pInputURL, strErrInfo);
			iRet = ERR_OPEN_MEDIA_FAIL;
			iErr = 1;
			break;
		}

		if ((iRet = avformat_find_stream_info(pIfmtCtx, 0)) < 0)
		{
			memset(strErrInfo, 0, 1024);
			av_strerror(iRet, strErrInfo, 1024);
			LOGFMTE("Failed to retrieve input stream information for %s , error info:%s\n", pInputURL, strErrInfo);
			iErr = 1;
			iRet = ERR_CANNOT_FIND_VIDEO;
			break;
		}

		if (pMediaConv->pOutputPrivate == NULL && pOutputURL != NULL && strlen(pOutputURL) > 0 && pOutputFmt != NULL && strlen(pOutputFmt) > 0)
		{
			iRet = avformat_alloc_output_context2(&pOfmtCtx, NULL, pOutputFmt, pOutputURL);
			if (iRet < 0)
			{
				LOGE("Could not create output context\n");
				iRet = ERR_OPEN_MEDIA_FAIL;
				break;
			}
		}

		for (iIndex = 0; iIndex < pIfmtCtx->nb_streams; iIndex++)
		{
			pInputStream = pIfmtCtx->streams[iIndex];
			switch (pInputStream->codecpar->codec_type)
			{
			case AVMEDIA_TYPE_VIDEO:
			{
				iInputVideoIndex = iIndex;
				if (pPrivate == NULL)
				{
					iRet = SetVideoCodec(pOfmtCtx, pInputStream->codecpar->codec_id, pInputStream->codecpar->extradata, pInputStream->codecpar->extradata_size,
						pInputStream->codecpar->width, pInputStream->codecpar->height, &pVideoStream, &iOutputStreamCount, &iOutputVideoIndex);
					if (iRet != 0)
					{
						LOGE("Add Video Stream Error!\n");
					}
				}
				else
				{
					//Parse Video Header
					if (pInputStream->codecpar->codec_id == AV_CODEC_ID_H264 && pInputStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
					{
						if (pInputStream->codecpar->extradata != NULL && pInputStream->codecpar->extradata_size > 4 && memcmp(pInputStream->codecpar->extradata, aSync, 4) != 0)
						{
							iParsed = qcAV_ConvertAVCNalHead(aVideoHeader, iVideoHeaderSize, pInputStream->codecpar->extradata, pInputStream->codecpar->extradata_size, iNalLengthSize);
							pMediaConv->iNalLengthSize = iNalLengthSize;
						}
					}
				}
				break;
			}
			case AVMEDIA_TYPE_AUDIO:
			{
				iInputAudioIndex = iIndex;
				if (pPrivate == NULL)
				{
					iRet = SetAudioCodec(pOfmtCtx, pInputStream->codecpar->codec_id, pInputStream->codecpar->extradata, pInputStream->codecpar->extradata_size,
						pInputStream->codecpar->channels, pInputStream->codecpar->sample_rate, &pAudioStream, &iOutputStreamCount, &iOutputAudioIndex);
					if (iRet != 0)
					{
						LOGE("Add Audio Stream Error!\n");
					}
				}
				else
				{
					//Parse Audio Header
					if (pInputStream->codecpar->codec_id == AV_CODEC_ID_AAC && pInputStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
					{

					}
				}
				break;
			}

			default:
			{
				break;
			}
			}
		}

		if (pOfmtCtx != NULL)
		{
			iRet = PrepareOuptput(pOfmtCtx, pOutputURL);
			if (iRet != 0)
			{
				iRet = ERR_OPEN_MEDIA_FAIL;
				break;
			}
		}

		//Everything is done
		strcpy(pMediaConv->strInputURL, pInputURL);
		pMediaConv->pIfmtCtx = pIfmtCtx;
		pMediaConv->iInputVideoIdx = iInputVideoIndex;
		pMediaConv->iInputAudioIdx = iInputAudioIndex;
		pMediaConv->pOutputPrivate = pPrivate;
		pMediaConv->iIndexForPrivate = iIndexForPrivate;
		pMediaConv->iIndexForPrivateSub = iIndexForPrivateSub;

		if (pMediaConv->pOutputPrivate == NULL)
		{
			strcpy(pMediaConv->strOutputFmt, pOutputFmt);
			strcpy(pMediaConv->strOutputURL, pOutputURL);
			pMediaConv->pOfmtCtx = pOfmtCtx;
			pMediaConv->iOutputVideoIdx = iOutputVideoIndex;
			pMediaConv->iOutputAudioIdx = iOutputAudioIndex;
		}

		iErr = 0;
		iRet = 0;
	} while (0);

	if (iErr != 0 || iRet < 0)
	{
		if (pIfmtCtx != NULL)
		{
			avformat_close_input(&pIfmtCtx);
			avformat_free_context(pIfmtCtx);
		}

		if (pOfmtCtx != NULL)
		{
			avformat_close_input(&pOfmtCtx);
			avformat_free_context(pOfmtCtx);
		}
	}

	return iRet;
}


int SetPrivateOutput(S_Media_Conv*   pMediaConv, void*  pPrivate)
{
	int iRet = 0;

	do 
	{
		if (pMediaConv == NULL)
		{
			iRet = ERR_INVALID_PARAMETERS;
			break;
		}

		pMediaConv->pOutputPrivate = pPrivate;
	} while (0);

	return iRet;
}


int DoMediaConv(S_Media_Conv* pMediaConv)
{
	int iRet = 0;
	AVPacket pkt;
	AVFormatContext *pIfmtCtx = NULL;
	AVFormatContext *pOfmtCtx = NULL;
	void*            pPrivateOutput = NULL;
	AVStream*        pStream = NULL;
	char             strInput[1024] = { 0 };
	char             strOutput[1024] = { 0 };
	char             strOutputFmt[1024] = { 0 };
	int              iIndexForPrivate = -1;
	int              iIndexForPrivateSub = -1;
	int iOutputStreamIndex = -1;
	int               iReOpenValue = 0;

	do
	{
		if (pMediaConv == NULL)
		{
			iRet = ERR_INVALID_PARAMETERS;
			break;
		}

		pIfmtCtx = pMediaConv->pIfmtCtx;
		pOfmtCtx = pMediaConv->pOfmtCtx;
		pPrivateOutput = pMediaConv->pOutputPrivate;
		if (pIfmtCtx == NULL || (pOfmtCtx == NULL && pPrivateOutput == NULL))
		{
			iRet = ERR_INVALID_PARAMETERS;
			break;
		}

		strcpy(strInput, pMediaConv->strInputURL);
		strcpy(strOutput, pMediaConv->strOutputURL);
		strcpy(strOutputFmt, pMediaConv->strOutputFmt);
		iIndexForPrivate = pMediaConv->iIndexForPrivate;
		iIndexForPrivateSub = pMediaConv->iIndexForPrivateSub;
		if (pMediaConv->iRunFlag == 0)
		{
			pMediaConv->iRunFlag = 1;
		}

		while (pMediaConv->iRunFlag == 1)
		{
			av_init_packet(&pkt);
			iOutputStreamIndex = -1;
			iRet = av_read_frame(pIfmtCtx, &pkt);
			if (iRet < 0)
			{
				LOGI("read frame fail!, reset all context\n");
				UnInitMediaConv(pMediaConv);
				iReOpenValue = InitMediaConv(pMediaConv, strInput, strOutput, strOutputFmt, pPrivateOutput, iIndexForPrivate, iIndexForPrivateSub);
				while (iReOpenValue != 0)
				{
					LOGI("InitMediaConv Fail!");
					iReOpenValue = InitMediaConv(pMediaConv, strInput, strOutput, strOutputFmt, pPrivateOutput, iIndexForPrivate, iIndexForPrivateSub);
#ifdef _LINUX_
					usleep(5000);
#endif

#ifdef _WIN32
					Sleep(5);
#endif
				}
				pMediaConv->iRunFlag = 1;
				pIfmtCtx = pMediaConv->pIfmtCtx;
				pOfmtCtx = pMediaConv->pOfmtCtx;
				continue;
			}

			if (pOfmtCtx != NULL)
			{
				if (pkt.stream_index == pMediaConv->iInputVideoIdx)
				{
					iOutputStreamIndex = pMediaConv->iOutputVideoIdx;
				}

				if (pkt.stream_index == pMediaConv->iInputAudioIdx)
				{
					iOutputStreamIndex = pMediaConv->iOutputAudioIdx;
				}

				if (iOutputStreamIndex == -1)
				{
					continue;
				}
				pStream = pOfmtCtx->streams[iOutputStreamIndex];

				pkt.pts = av_rescale_q_rnd(pkt.pts, pIfmtCtx->streams[pkt.stream_index]->time_base, pStream->time_base, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
				pkt.dts = av_rescale_q_rnd(pkt.dts, pIfmtCtx->streams[pkt.stream_index]->time_base, pStream->time_base, (enum AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));

				pkt.pos = -1;
				pkt.stream_index = iOutputStreamIndex;

				if (iOutputStreamIndex == pMediaConv->iOutputVideoIdx)
				{
					//printf("video time: pts:%lld, dts:%lld\n", pkt.pts, pkt.dts);
				}

				if (iOutputStreamIndex == pMediaConv->iOutputAudioIdx)
				{
					//printf("audio time: pts:%lld, dts:%lld\n", pkt.pts, pkt.dts);
				}


				iRet = av_write_frame(pOfmtCtx, &pkt);
				if (iRet < 0)
				{
					LOGE("Error muxing packet\n");
				}
			}

			if (pPrivateOutput != NULL)
			{
				TransactMediaFrameFromFFmpeg(&pkt, pMediaConv, pPrivateOutput, pMediaConv->iIndexForPrivate, pMediaConv->iIndexForPrivateSub);
			}


			av_free_packet(&pkt);
		}
	} while (0);

	return iRet;
}


int PeekStreamURL(char*  pInputURL, void*  pMediaInfoPriv)
{
	int iRet = 0;
	S_STREAM_ITEM_INFO*   pMediaInfo = (S_STREAM_ITEM_INFO*)pMediaInfoPriv;
	AVFormatContext *pIfmtCtx = NULL;
	AVStream*        pStream = NULL;
	AVDictionary*     pOptions = NULL;
	int               iTimeOutForLiveStream = 0;
	char              strTime[128] = { 0 };
	char              strErrInfo[1024] = { 0 };
	int               iErr = 0;
	int               iIndex = 0;

	do 
	{
		if (strlen(pInputURL) > 7 && memcmp(pInputURL, "rtsp://", 7) == 0)
		{
			av_dict_set(&pOptions, "rtsp_flags", "prefer_tcp", 0);

			if (iTimeOutForLiveStream <= 0)
			{
				iTimeOutForLiveStream = 5;
			}
			sprintf(strTime, "%lld", (long long)iTimeOutForLiveStream * 1000000);
			av_dict_set(&pOptions, "stimeout", strTime, 0);
		}

		av_dict_set(&pOptions, "fflags", "nobuffer", 0);


		//Open Input URL
		if ((iRet = avformat_open_input(&pIfmtCtx, pInputURL, 0, &pOptions)) < 0)
		{
			memset(strErrInfo, 0, 1024);
			av_strerror(iRet, strErrInfo, 1024);
			LOGFMTE("Could not open input file: %s , error info:%s\n", pInputURL, strErrInfo);
			iRet = ERR_OPEN_MEDIA_FAIL;
			iErr = 1;
			break;
		}

		if ((iRet = avformat_find_stream_info(pIfmtCtx, 0)) < 0)
		{
			memset(strErrInfo, 0, 1024);
			av_strerror(iRet, strErrInfo, 1024);
			LOGFMTE("Failed to retrieve input stream information for %s \n", pInputURL, strErrInfo);
			iErr = 1;
			iRet = ERR_CANNOT_FIND_VIDEO;
			break;
		}


		for (iIndex = 0; iIndex < pIfmtCtx->nb_streams; iIndex++)
		{
			pStream = pIfmtCtx->streams[iIndex];
			switch (pStream->codecpar->codec_type)
			{
				case AVMEDIA_TYPE_VIDEO:
				{
					pMediaInfo->iVideoCodec = (int)pStream->codecpar->codec_id;
					pMediaInfo->iWidth = pStream->codecpar->width;
					pMediaInfo->iHeight = pStream->codecpar->height;
					if (pStream->codecpar->extradata != NULL && pStream->codecpar->extradata_size < 128)
					{
						memcpy(pMediaInfo->uVideoExtra, pStream->codecpar->extradata, pStream->codecpar->extradata_size);
						pMediaInfo->iVideoExtraSize = pStream->codecpar->extradata_size;
					}
					break;
				}
				case AVMEDIA_TYPE_AUDIO:
				{
					pMediaInfo->iAudioCodec = (int)pStream->codecpar->codec_id;
					pMediaInfo->iAudioSampleRate = pStream->codecpar->sample_rate;
					pMediaInfo->iAudioChannel = pStream->codecpar->channels;
					pMediaInfo->iAudioBitWidth = pStream->codecpar->bits_per_coded_sample;
					if (pStream->codecpar->extradata != NULL && pStream->codecpar->extradata_size < 128)
					{
						memcpy(pMediaInfo->uAudioExtra, pStream->codecpar->extradata, pStream->codecpar->extradata_size);
						pMediaInfo->iAudioExtraSize = pStream->codecpar->extradata_size;
					}
					break;
				}

				default:
				{
					break;
				}
			}
		}

	} while (0);


	if (pIfmtCtx != NULL)
	{
		avformat_close_input(&(pIfmtCtx));
		avformat_free_context(pIfmtCtx);
	}

	return iRet;
}

int UnInitMediaConv(S_Media_Conv*   pMediaConv)
{
	int iRet = 0;

	do 
	{
		if (pMediaConv == NULL)
		{
			break;
		}

		if (pMediaConv->pIfmtCtx != NULL)
		{
			avformat_close_input(&(pMediaConv->pIfmtCtx));
			avformat_free_context(pMediaConv->pIfmtCtx);
		}

		if (pMediaConv->pOfmtCtx != NULL)
		{
			avformat_close_input(&(pMediaConv->pOfmtCtx));
			avformat_free_context(pMediaConv->pOfmtCtx);
		}
	} while (0);


	pMediaConv->pIfmtCtx = pMediaConv->pOfmtCtx = NULL;
	return iRet;
}

void* ThreadFuncForMediaConv(void*   pMediaConv)
{
	int iRet = 0;
	S_Media_Conv*    pMediaConvIns = (S_Media_Conv*)pMediaConv;

	do 
	{
		if (pMediaConvIns == NULL)
		{
			break;
		}

		iRet = DoMediaConv(pMediaConvIns);
	} while (0);

	return NULL;
}