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


#include "PlayM4.h"
#include "stdio.h"
#include "string.h"

#define BUILD_INDEX_TIMEOUT  500
#define NO_FRAME_TIMEOUT     20


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
#define ERR_OPEN_VIDEO_DECODER_FAIL                 0x8000000b




char   g_strDumpBase[512] = { 0 };
int   g_iRefDone = 0;
int   g_iFileEndReach = 0;
int   g_iGetFrameDone = 0;
int   g_iCurFrameCount = 0;
int   g_iStartTime = 0;
int   g_iEndTime = 0;
int   g_iLastDumpTime = 0;
int   g_iInterval = 0;




int  ParseCmd(int argc, char* argv[], int* piInterval, char**  ppInputURL, int* piSeekStartPos, int* piSeekEndPos, char** ppOutputPrefix);
int  RunFrameExtract(char* pFilePath, int iSeekStartPos, int iInterval, int iSeekEndPos, char* pBasePath);
int  NeedDump(int  iTimeStart, int iTimeEnd, int iTimeInput, int iLastTime, int ilInterval);

#ifdef WIN32
void CALLBACK FileEndCallback(long   nPort, void        *pUser)
{
        g_iFileEndReach = 1;
        printf("file end reach!\n");
}
#endif


#ifdef __linux__
void CALLBACK FileEndCallback(int   nPort, void        *pUser)
{
        g_iFileEndReach = 1;
        printf("file end reach!\n");
}
#endif

void cm_Sleep(int iMillSec);

void CALLBACK DisplayCBFun(DISPLAY_INFO   *pstDisplayInfo)
{
	int  iRet = 0;
	char  strDumpPath[1024]= {0};

	switch (pstDisplayInfo->nType)
	{
		case T_YV12:
		case T_UYVY:
		case T_RGB32:
		{
			if (pstDisplayInfo->nStamp > g_iEndTime)
			{
				g_iGetFrameDone = 1;
			}

			if (NeedDump(g_iStartTime, g_iEndTime, pstDisplayInfo->nStamp, g_iLastDumpTime, g_iInterval) == 1)
			{
				g_iLastDumpTime = pstDisplayInfo->nStamp;
				sprintf(strDumpPath, "%s%d.jpg", g_strDumpBase, pstDisplayInfo->nStamp);
				printf("time:%d\n", pstDisplayInfo->nStamp);
				iRet = PlayM4_ConvertToJpegFile(pstDisplayInfo->pBuf, pstDisplayInfo->nBufLen, pstDisplayInfo->nWidth, pstDisplayInfo->nHeight, pstDisplayInfo->nType, strDumpPath);
				if (iRet == 0)
				{
					printf("error:%d\n", PlayM4_GetLastError(pstDisplayInfo->nPort));
				}
			}
			g_iCurFrameCount++;
			break;
		}
	}
}

#ifdef WIN32
void CALLBACK FileRefDoneCB(DWORD nPort, DWORD nUser)
{
	g_iRefDone = 1;
}
#endif


#ifdef __linux__
void CALLBACK FileRefDoneCB(DWORD nPort, void* pUser)
{
        g_iRefDone = 1;
}
#endif



int main(int argc, char* argv[])
{
	int iRet = 0;
	int iSeekPos = 0;
	int iInterval = 0;
	int  iSeekStartPos = 0;
	int  iSeekEndPos = 0;
	char*   pIntputURL = NULL;
	char*   pOutputURLPrefix = NULL;

	iRet = ParseCmd(argc, argv, &iInterval, &pIntputURL, &iSeekStartPos, &iSeekEndPos, &pOutputURLPrefix);
	if (iRet != 0)
	{
		return ERR_INVALID_PARAMETERS;
	}

	iRet = RunFrameExtract(pIntputURL, iSeekStartPos, iInterval, iSeekEndPos, pOutputURLPrefix);

	return 0;
}


int  RunFrameExtract(char* pFilePath, int iSeekStartPos, int iInterval, int iSeekEndPos, char* pBasePath)
{
	int iRet = 0;
	BOOL bInnerRet = FALSE;
	LONG  iPort = 0;
	int   iBuildIndexWaitCount = 0;
	int   iDecodeWaitCount = 0;
	int   iLastFrameCount = 0xfffffff;
	int   iIndex = 0;
	do
	{
		if(pBasePath == NULL || strlen(pBasePath) == 0)
		{
			strcpy(g_strDumpBase, "./");
		}
		else
		{
			strcpy(g_strDumpBase, pBasePath);
		}
		
		bInnerRet = PlayM4_GetPort(&iPort);
		if (bInnerRet == FALSE)
		{
			printf("GetPort error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}


		g_iStartTime = iSeekStartPos;
		g_iEndTime = iSeekEndPos;
		g_iInterval = iInterval;


		bInnerRet  = PlayM4_SetFileRefCallBack(iPort, FileRefDoneCB, NULL);
		if (bInnerRet == FALSE)
		{
			printf("SetFileRefCallBack  error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}

		bInnerRet = PlayM4_SetFileEndCallback(iPort, FileEndCallback, NULL);
		if (bInnerRet == FALSE)
		{
			printf("SetFileRefCallBack  error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}

#ifdef WIN32
                bInnerRet = PlayM4_SetDisplayCallBackEx(iPort, DisplayCBFun, (long)iPort);
#endif


#ifdef __linux__
                bInnerRet = PlayM4_SetDisplayCallBackEx(iPort, DisplayCBFun, (void*)iPort);
#endif

		if (bInnerRet == FALSE)
		{
			printf("SetDisplayCallBackEx  error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}
		
		bInnerRet = PlayM4_OpenFile(iPort, pFilePath);
		if (bInnerRet == FALSE)
		{
			printf("OpenFile  error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}


		bInnerRet = PlayM4_Play(iPort, NULL);
		if (bInnerRet == FALSE)
		{
			printf("Play  error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}

                for (iIndex = 0; iIndex < 4; iIndex++)
                {
                        bInnerRet = PlayM4_Fast(iPort);
                        if (bInnerRet == FALSE)
                        {
                                printf("PlayM4_Fast, error:%d\n", PlayM4_GetLastError(iPort));
                                iRet = 1;
                                break;
                        }
                }


		while (1)
		{
			if (g_iRefDone == 0)
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


		if (g_iRefDone == 0)
		{
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}


		bInnerRet = PlayM4_SetPlayedTimeEx(iPort, g_iStartTime);
		if (bInnerRet == FALSE)
		{
			printf("SetPlayedTime error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}

		while (1)
		{

			if (g_iFileEndReach == 1 || g_iGetFrameDone == 1)
			{
				break;
			}

			if (g_iCurFrameCount != iLastFrameCount)
			{
				iLastFrameCount = g_iCurFrameCount;
				iDecodeWaitCount = 0;
			}
			else
			{
				iDecodeWaitCount++;
			}

			if (iDecodeWaitCount > NO_FRAME_TIMEOUT)
			{
				break;
			}

			cm_Sleep(50);
		}

		iRet = PlayM4_Stop(iPort);
		if (iRet == 0)
		{
			printf("error:%d\n", PlayM4_GetLastError(iPort));
			iRet = ERR_OPEN_MEDIA_FAIL;
			break;
		}
	} while (0);


	PlayM4_CloseFile(iPort);
	PlayM4_FreePort(iPort);
	return 0;
}

int  ParseCmd(int argc, char* argv[], int* piInterval, char**  ppInputURL, int* piSeekStartPos, int* piSeekEndPos, char** ppOutputPrefix)
{
	int  iRet = 0;
	int  iIndex = 0;
	char*  pCurOpt = NULL;
	float  fRate = 0;
	*piSeekStartPos = 0;
	*piSeekEndPos = 10;
	*piInterval = 0;

	if (argc < 11)
	{
		printf("invalid parameter!\n");
		printf("-i input_url     set the input url\n"
			"-ss start_time_off        set the start time offset\n"
			"-to end_time_off          set the end time offset\n"
			"-r rate             set frame rate(Hz value, fraction or abbreviation)\n"
			"-prefix output prefix           set the prefix for output\n"
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

		if (strcmp("-r", pCurOpt) == 0)
		{
			iRet = sscanf(argv[++iIndex], "%f", &fRate);
			if (iRet == 1 && fRate != 0)
			{
				*piInterval = (int)(1 / fRate * 1000);
			}
		}

		if (strcmp("-prefix", pCurOpt) == 0)
		{
			*ppOutputPrefix = argv[++iIndex];
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

int  NeedDump(int  iTimeStart, int iTimeEnd, int iTimeInput, int iLastTime, int iInterval)
{
	int iRet = 0;
	if (iTimeStart < iTimeInput && iTimeInput < iTimeEnd)
	{
		if (iLastTime == 0 || ((iTimeInput - iLastTime + 1) > iInterval))
		{
			iRet = 1;
		}
	}

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

