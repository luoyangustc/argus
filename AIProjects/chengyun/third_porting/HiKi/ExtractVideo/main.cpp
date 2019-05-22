#ifdef __linux__
#include "unistd.h"

#ifndef LONG
typedef int LONG;
#endif

#ifndef DWORD
typedef unsigned int DWORD;
#endif

#ifndef BOOL
typedef int BOOL;
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#endif

#ifdef WIN32
#include "windows.h"
#endif



void cm_Sleep(int iMillSec);

#include "PlayM4.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "string.h"
#include "stdint.h"
#include "x264.h"

#define BUILD_INDEX_TIMEOUT  100
#define NO_FRAME_TIMEOUT     100


#define ERR_INVALID_PARAMETERS    0x80000001
#define ERR_CANNOT_FIND_VIDEO              0x80000002
#define ERR_CANNOT_FIND_VIDEO_DECODER              0x80000003
#define ERR_CANNOT_OPEN_VIDEO_DECODER              0x80000004
#define ERR_CANNOT_OPEN_VIDEO_SCALE                0x80000005
#define ERR_CANNOT_OPEN_MJPEG                      0x80000006
#define ERR_CANNOT_OPEN_MJPEG_ENCODER              0x80000007
#define ERR_CANNOT_ALLOC_MEMORY                    0x80000008
#define ERR_ENCODING_JPG                           0x80000009
#define ERR_OPEN_MEDIA_FAIL                        0x8000000a
#define ERR_OPEN_VIDEO_DECODER_FAIL                0x8000000b
#define ERR_OPEN_DUMP_FILE_FAIL                    0x8000000c



typedef struct
{
	int iWidth;
	int iHeight;
	int iPts;
	int iFrameRate;

	int iNal;
	x264_nal_t* pNals;
	x264_t* pHandle;
	x264_picture_t* pPic_in;
	x264_picture_t* pPic_out;
	x264_param_t* pParam;
	FILE*   pH264Dump;
	int     iInitDone;
	char    strTmpFilePath[1024];
} S_H264_Enc_Ins;

char   gstrDumpFile[256] = { 0 };
bool   g_bRefDone = false;
bool   g_bFileEndReach = false;
bool   g_bGetFrameDone = false;
int    g_iCurFrameCount = 0;
int    g_iLastFrameCount = 0;
long   g_iStartTime = 0;
long   g_iEndTime = 0;
long   g_ulLastDumpTime = 0;
long   g_ulInterval = 0;
unsigned int g_iFrameRate = 0;
int g_iFmtForH264 = 0;

S_H264_Enc_Ins   g_sH264EncIns;
time_t   g_tStartTime = 0;
time_t   g_tEndTime = 0;


int  InitH264Enc(int iWidth, int iHeight, int iRate, int iFmtForX264, S_H264_Enc_Ins*  pEncIns);
int  EncVideoToH264(unsigned char* pData, int iSize, S_H264_Enc_Ins*  pEncIns);
int  FlushH264Enc(S_H264_Enc_Ins*  pEncIns);
int  UnInitH264Enc(S_H264_Enc_Ins*  pEncIns);


int  ParseCmd(int argc, char* argv[], char**  ppInputURL, int* piSeekStartPos, int* piSeekEndPos, char** ppOutputURL);
int  RunFrameExtract(char* pFilePath, int iSeekStartPos, int iSeekEndPos, char* pOutputURL);
int  NeedEnc(long  ulTimeStart, long ulTimeEnd, long ulTimeInput);
int  GetUniqueString(char*  pInputStringBuf);
int  GetX264CSPValue(int iPixFmtFromDev);
int  ConvertFileByFFmpeg(char*  pH264File, char* pOutputFile);

int  InitH264Enc(int iWidth, int iHeight, int iRate, int iFmtForX264, S_H264_Enc_Ins*  pEncIns)
{
	int iRet = 0;
	char    strUnique[512] = { 0 };

	memset(&g_sH264EncIns, 0, sizeof(S_H264_Enc_Ins));
	GetUniqueString(strUnique);

	do 
	{
		pEncIns->pH264Dump = fopen(strUnique, "wb");
		if (pEncIns->pH264Dump == NULL)
		{
			printf("create tmp file fail!\n");
			iRet = ERR_OPEN_DUMP_FILE_FAIL;
			break;
		}

		strcpy(pEncIns->strTmpFilePath, strUnique);
		
		
		pEncIns->pPic_in = (x264_picture_t*)malloc(sizeof(x264_picture_t));
		pEncIns->pPic_out = (x264_picture_t*)malloc(sizeof(x264_picture_t));
		pEncIns->pParam = (x264_param_t*)malloc(sizeof(x264_param_t));

		if (pEncIns->pPic_in == NULL || pEncIns->pPic_out == NULL || pEncIns->pParam == NULL)
		{
			iRet = ERR_CANNOT_ALLOC_MEMORY;
			break;
		}

		pEncIns->pParam->i_fps_den = 1;
		pEncIns->pParam->i_fps_num = g_iFrameRate;
		pEncIns->pParam->i_timebase_den = pEncIns->pParam->i_fps_num;
		pEncIns->pParam->i_timebase_num = pEncIns->pParam->i_fps_den;

		x264_param_default(pEncIns->pParam);
		pEncIns->pParam->i_width = iWidth;
		pEncIns->pParam->i_height = iHeight;
		pEncIns->pParam->i_sync_lookahead = 10;
		pEncIns->pParam->i_bframe = 0;

		pEncIns->pParam->i_csp = iFmtForX264;
		x264_param_apply_profile(pEncIns->pParam, x264_profile_names[3]);

		pEncIns->pHandle = x264_encoder_open(pEncIns->pParam);

		x264_picture_init(pEncIns->pPic_out);
		iRet = x264_picture_alloc(pEncIns->pPic_in, pEncIns->pParam->i_csp, pEncIns->pParam->i_width, pEncIns->pParam->i_height);
		if (iRet != 0)
		{
			iRet = ERR_CANNOT_ALLOC_MEMORY;
			break;
		}

		pEncIns->iInitDone = 1;
	} while (0);

	return 0;
}

int  EncVideoToH264(unsigned char* pData, int iSize, S_H264_Enc_Ins*  pEncIns)
{
	int iYSize = pEncIns->pParam->i_width*pEncIns->pParam->i_height;
	int iRet = 0;
	int iOffset = 0;
	int iIndex = 0;


	do 
	{
		if (pEncIns->iInitDone != 1)
		{
			break;
		}

		memcpy(pEncIns->pPic_in->img.plane[0], pData + iOffset, iYSize);
		iOffset += iYSize;
		memcpy(pEncIns->pPic_in->img.plane[2], pData + iOffset, iYSize / 4);
		iOffset += iYSize / 4;
		memcpy(pEncIns->pPic_in->img.plane[1], pData + iOffset, iYSize / 4);

		pEncIns->pPic_in->i_pts = pEncIns->iPts;

		iRet = x264_encoder_encode(pEncIns->pHandle, &pEncIns->pNals, &pEncIns->iNal, pEncIns->pPic_in, pEncIns->pPic_out);
		if (iRet < 0)
		{
			printf("Error.\n");
			return -1;
		}


		if (pEncIns->pH264Dump != NULL)
		{
			//printf("cur g_iNalCount:%d\n", pEncIns->iNal);
			for (iIndex = 0; iIndex < pEncIns->iNal; iIndex++)
			{
				fwrite(pEncIns->pNals[iIndex].p_payload, 1, pEncIns->pNals[iIndex].i_payload, pEncIns->pH264Dump);
			}
		}
	} while (0);



	return 0;
}


int  FlushH264Enc(S_H264_Enc_Ins*  pEncIns)
{
	int j = 0;
	int iRet = 0;

	do 
	{
		if (pEncIns->iInitDone != 1)
		{
			break;
		}

		while (1)
		{
			iRet = x264_encoder_encode(pEncIns->pHandle, &pEncIns->pNals, &pEncIns->iNal, NULL, pEncIns->pPic_out);
			if (iRet == 0)
			{
				break;
			}

			//printf("Flush 1 frame.\n");
			if (pEncIns->pH264Dump != NULL)
			{
				for (j = 0; j < pEncIns->iNal; ++j)
				{
					fwrite(pEncIns->pNals[j].p_payload, 1, pEncIns->pNals[j].i_payload, pEncIns->pH264Dump);
				}
			}
		}

	} while (0);

	return 0;
}

int  UnInitH264Enc(S_H264_Enc_Ins*  pEncIns)
{
	int iRet = 0;

	do 
	{
		if (pEncIns->iInitDone != 1)
		{
			break;
		}

		if (pEncIns->pH264Dump != NULL)
		{
			fclose(pEncIns->pH264Dump);
			pEncIns->pH264Dump = NULL;
		}

		x264_picture_clean(pEncIns->pPic_in);
		x264_encoder_close(pEncIns->pHandle);
		pEncIns->pHandle = NULL;

		free(pEncIns->pPic_in);
		free(pEncIns->pPic_out);
		free(pEncIns->pParam);

	} while (0);

	return 0;
}


#ifdef WIN32
void CALLBACK FileEndCallback(long   nPort, void        *pUser)
{
	g_bFileEndReach = true;
	printf("file end reach!\n");
}
#endif


#ifdef __linux__
void CALLBACK FileEndCallback(int   nPort, void        *pUser)
{
	g_bFileEndReach = true;
	printf("file end reach!\n");
}
#endif


void CALLBACK DisplayCBFun(DISPLAY_INFO   *pstDisplayInfo)
{
	bool bRet = false;
	int  iRet = 0;
	time_t  tRet = 0;

	switch (pstDisplayInfo->nType)
	{
		case T_YV12:
		{
			if (pstDisplayInfo->nStamp > g_iEndTime)
			{
				g_bGetFrameDone = true;
			}

			if (NeedEnc(g_iStartTime, g_iEndTime, pstDisplayInfo->nStamp) == 1)
			{
				//printf("pstDisplayInfo->nStamp:%d\n", pstDisplayInfo->nStamp);  
				if (g_sH264EncIns.iPts == 0)
				{
					tRet = time(&g_tStartTime);
					iRet = InitH264Enc(pstDisplayInfo->nWidth, pstDisplayInfo->nHeight, g_iFrameRate, X264_CSP_I420, &g_sH264EncIns);
					if(iRet != 0)
					{
						printf("Init H264 Enc Fail!\n");
					}
				}
				//printf("time:%d\n", pstDisplayInfo->nStamp);

				EncVideoToH264((unsigned char*)pstDisplayInfo->pBuf, pstDisplayInfo->nBufLen, &g_sH264EncIns);
				g_sH264EncIns.iPts++;
				g_iCurFrameCount++;
			}

			if (g_bGetFrameDone == true)
			{
				FlushH264Enc(&g_sH264EncIns);
				tRet = time(&g_tEndTime);
				printf("Use time for encoding:%d second\n", g_tEndTime - g_tStartTime);
			}

			break;
		}
	}
}

#ifdef WIN32
void CALLBACK FileRefDoneCB(DWORD nPort, DWORD nUser)
{
	g_bRefDone = true;
}
#endif


#ifdef __linux__
void CALLBACK FileRefDoneCB(DWORD nPort, void* pUser)
{
	g_bRefDone = true;
}
#endif


int main(int argc, char* argv[])
{
	int iRet = 0;
	int iSeekStartPos = 0;
	int iSeekEndPos = 0;
	char*   pIntputURL = NULL;
	char*   pOutputURL = NULL;

	iRet = ParseCmd(argc, argv, &pIntputURL, &iSeekStartPos, &iSeekEndPos, &pOutputURL);
	if (iRet != 0)
	{
		return ERR_INVALID_PARAMETERS;
	}

	memset(&g_sH264EncIns, 0, sizeof(S_H264_Enc_Ins));
	iRet = RunFrameExtract(pIntputURL, iSeekStartPos, iSeekEndPos, pOutputURL);

	return 0;
}



int  RunFrameExtract(char* pFilePath, int iSeekStartPos, int iSeekEndPos, char* pOutputURL)
{
	int  iRet = 0;
	BOOL bRetInner = FALSE;
	LONG  nPort = 0;
	bool bRet = true;
	int   iBuildIndexWaitCount = 0;
	int   iDecodeWaitCount = 0;
	int  ulLastFrameCount = 0xfffffff;
	int   dwFrameRate = 0;
	time_t   tRet = 0;
	int    iIndex = 0;
	do
	{
		tRet = time(&g_tStartTime);
		bRetInner = PlayM4_GetPort(&nPort);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		g_iStartTime = iSeekStartPos;
		g_iEndTime = iSeekEndPos;
		
		printf("start_time:%d, end_time:%d\n", g_iStartTime, g_iEndTime);
		bRetInner = PlayM4_SetFileRefCallBack(nPort, FileRefDoneCB, 0);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		bRetInner = PlayM4_SetFileEndCallback(nPort, FileEndCallback, NULL);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		bRetInner = PlayM4_OpenFile(nPort, pFilePath);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}


#ifdef WIN32
		bRetInner = PlayM4_SetDisplayCallBackEx(nPort, DisplayCBFun, (long)nPort);
#endif


#ifdef __linux__
		bRetInner = PlayM4_SetDisplayCallBackEx(nPort, DisplayCBFun, (void*)nPort);
#endif
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		bRetInner = PlayM4_Play(nPort, NULL);
		if (bRetInner == 0)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}


		for (iIndex = 0; iIndex < 2; iIndex++)
		{
			bRetInner = PlayM4_Fast(nPort);
			if (bRetInner == 0)
			{
				printf("PlayM4_Fast, error:%d\n", PlayM4_GetLastError(nPort));
				iRet = 1;
				break;
			}
		}


		while (1)
		{
			if (g_bRefDone == false)
			{
				cm_Sleep(50);
				iBuildIndexWaitCount++;
				if (iBuildIndexWaitCount < BUILD_INDEX_TIMEOUT)
				{
					continue;
				}
				else
				{
					break;
				}
			}
			else
			{
				break;
			}
		}


		if (g_bRefDone == false)
		{
			break;
		}

		tRet = time(&g_tEndTime);
		printf("Use time for Index build:%d second\n", g_tEndTime - g_tStartTime);
		g_tStartTime = g_tEndTime;

		g_iFrameRate = PlayM4_GetCurrentFrameRate(nPort);
		if (g_iFrameRate == 0xffffffff)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		printf("frame rate:%d\n", g_iFrameRate);

		bRetInner = PlayM4_SetPlayedTimeEx(nPort, g_iStartTime);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}

		tRet = time(&g_tEndTime);
		printf("Use time for Seeking:%d second\n", g_tEndTime - g_tStartTime);
		g_tStartTime = g_tEndTime;

		while (1)
		{

			if (g_bFileEndReach == true || g_bGetFrameDone == true)
			{
				break;
			}

			if (g_iCurFrameCount != ulLastFrameCount)
			{
				ulLastFrameCount = g_iCurFrameCount;
				iDecodeWaitCount = 0;
			}
			else
			{
				iDecodeWaitCount++;
			}

			if (iDecodeWaitCount > NO_FRAME_TIMEOUT)
			{
				printf("no frame come! time out!\n");
				break;
			}

			cm_Sleep(50);
		}

		bRetInner = PlayM4_Stop(nPort);
		if (bRetInner == FALSE)
		{
			printf("error:%d\n", PlayM4_GetLastError(nPort));
			iRet = 1;
			break;
		}
	} while (0);

	printf("file end:%d, get frame done:%d\n", g_bFileEndReach, g_bGetFrameDone);
	if(g_bFileEndReach == true && g_bGetFrameDone == false)	
	{
		FlushH264Enc(&g_sH264EncIns);
		tRet = time(&g_tEndTime);
		printf("Use time for encoding:%d second\n", g_tEndTime - g_tStartTime);
	}
	
	PlayM4_CloseFile(nPort);
	UnInitH264Enc(&g_sH264EncIns);
	printf("Frame Count:%d\n", g_iCurFrameCount);
	ConvertFileByFFmpeg(g_sH264EncIns.strTmpFilePath, pOutputURL);
	remove(g_sH264EncIns.strTmpFilePath);
	return 0;
}

int  ParseCmd(int argc, char* argv[], char**  ppInputURL, int* piSeekStartPos, int* piSeekEndPos, char** ppOutputURL)
{
	int  iRet = 0;
	int  iIndex = 0;
	char*  pCurOpt = NULL;
	*piSeekStartPos = 0;
	*piSeekEndPos = 0x7fffffff;
	if (argc < 5)
	{
		printf("invalid parameter!\n");
		printf("-i input_url     set the input url\n"
			"-ss time_start      set the start time offset\n"
			"-to time_end        set the end time offset\n"
			"-o output_url       set the output url\n"
			);
		return ERR_INVALID_PARAMETERS;
	}

	while (iIndex < argc)
	{
		pCurOpt = argv[iIndex];
		if (strcmp("-i", pCurOpt) == 0)
		{
			*ppInputURL = argv[++iIndex];
		}

		if (strcmp("-ss", pCurOpt) == 0)
		{
			iRet = sscanf(argv[++iIndex], "%d", piSeekStartPos);
			*piSeekStartPos = *piSeekStartPos * 1000;
		}

		if (strcmp("-to", pCurOpt) == 0)
		{
			iRet = sscanf(argv[++iIndex], "%d", piSeekEndPos);
			*piSeekEndPos = *piSeekEndPos * 1000;
		}

		if (strcmp("-o", pCurOpt) == 0)
		{
			*ppOutputURL = argv[++iIndex];
		}

		if (iRet == -1)
		{
			printf("invalid parameter!\n");
			return ERR_INVALID_PARAMETERS;
		}
		iIndex++;
	}

	return 0;
}

int  NeedEnc(long  ulTimeStart, long ulTimeEnd, long ulTimeInput)
{
	int iRet = 0;
	if (ulTimeStart < ulTimeInput && ulTimeInput < ulTimeEnd)
	{
		iRet = 1;
	}

	return iRet;
}

int  GetUniqueString(char*  pInputStringBuf)
{
#ifdef __linux__
	pid_t pid = getpid();
	time_t tValue;
	tValue = time(NULL);

	sprintf(pInputStringBuf, "%d_%d.h264", (unsigned int)pid, tValue);

#endif

#ifdef _WINDOWS_
	time_t tValue;
	tValue = time(NULL);
	sprintf(pInputStringBuf, "%d.h264", tValue);
#endif

	return 0;
}

int  ConvertFileByFFmpeg(char*  pH264File, char* pOutputFile)
{
	int iRet = 0;
	char  strCmd[4096] = { 0 };
#ifdef WIN32
	sprintf(strCmd, "ffmpeg.exe -i %s -c copy -y %s", pH264File, pOutputFile);
#endif

#ifdef __linux__
	sprintf(strCmd, "./ffmpeg -i %s -c copy -y %s", pH264File, pOutputFile);
#endif

	iRet = system(strCmd);
	return iRet;
}


void cm_Sleep(int iMillSec)
{
#ifdef __linux__
	usleep(iMillSec*1000);
#endif

#ifdef WIN32
	Sleep(iMillSec);
#endif
}
