#include "HCNetSDK.h"
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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

#define DVR_STATE_FINDING                 1




typedef struct
{
	int iYear;
	int iMonth;
	int iDay;
	int iHour;
	int iMin;
	int iSec;
} S_Full_Time;

int AddTimeOneSec(S_Full_Time* pTimeOriginal, S_Full_Time* pTimeAdded);
int ParseCmd(int iArgc, char*  argv[], char**  ppIPCUid, char** ppInputStringTime, char** ppPrefix, char** ppDeviceIP, int* piChannelIdx, char ** ppDvrUId, char** ppDvrUPwd, int* piPort);
int LoadCfg(char*  pCfgFile, char*  pstrDeviceIP, int*  piDeviceIndex, char*  pstrUserId, char*  pstrUserPwd, int* piPort);

static void  RemoveLineSeps(char* pInput)
{
	char*    pCur = NULL;
	if (pInput != NULL)
	{
		pCur = pInput + strlen(pInput) - 1;
		while (*pCur == '\r' || *pCur == '\n')
		{
			*pCur = 0;
			pCur--;
		}
		return;
	}
}


int main(int argc, char* argv[])
{
	LONG  lFileHandle = 0;
	LONG lFind = 0;
	LONG   lUserId = 0;
	LONG   lRet = 0;
	LONG   lFindHandle = 0;
	int iRet = 0;
	BOOL  bRet = FALSE;
	DWORD   dwErr = 0;
	NET_DVR_INIT_CFG_ABILITY  sDvrIniCfg;
	NET_DVR_USER_LOGIN_INFO   sUserLogin;
	NET_DVR_DEVICEINFO_V40    sDvrDevInfo;

	NET_DVR_RECORD_TIME_SPAN_INQUIRY   sTimeSpanInquiry;
	NET_DVR_RECORD_TIME_SPAN    sTimeSpan;
	NET_DVR_FILECOND_V40        sFileCond;
	NET_DVR_FINDDATA_V40        sFindData;
	char*    pIPCUId = NULL;
	char*    pInputStringTime = NULL;
	char*    pPrefix = NULL;
	char*    pFileDvrArray[256] = {0};
	char*    pFilelocalArray[256] = { 0 };
	int      iFileCount = 0;
	char     strFileDumpPath[256] = { 0 };
	char     strDirPrefix[512] = {0};
	int      iIndex = 0;
	DWORD    dwSpeed = 2048;
	int      iPos = 0;
	int      iPort = 0;
	char*     pstrIP = NULL;
	char*     pstrUserId = NULL;
	char*     pstrUserPwd = NULL;
	int      iChannelIdx = 0;
	S_Full_Time    sBeginTime = {0};
	S_Full_Time    sEndTime = {0};
	int      iState = 0;

	do 
	{
		iRet = ParseCmd(argc, argv, &pIPCUId, &pInputStringTime, &pPrefix, &pstrIP, &iChannelIdx, &pstrUserId, &pstrUserPwd, &iPort);
		if(iRet != 0 || pstrIP == NULL || pstrUserId == NULL || pstrUserPwd == NULL)
		{
			printf("Invalid Parameters\n");
			break;
		}

		//Modidy the pPrefix
		strcat(strDirPrefix, pPrefix);
		if(strDirPrefix[strlen(pPrefix)-1] != '/')
		{
			strDirPrefix[strlen(pPrefix)] = '/';
		}

		//convert time
		sscanf(pInputStringTime, "%4d%02d%02d%02d%02d%02d", &sBeginTime.iYear, &sBeginTime.iMonth, &sBeginTime.iDay, &sBeginTime.iHour, &sBeginTime.iMin, &sBeginTime.iSec);

		printf("%d-%02d-%02d-%02d-%02d-%02d\n", sBeginTime.iYear, sBeginTime.iMonth, sBeginTime.iDay, sBeginTime.iHour, sBeginTime.iMin, sBeginTime.iSec);



		//int LoadCfg(char*  pCfgFile, char*  pstrDeviceIP, int*  piDeviceIndex, char*  pstrUserId, char*  pstrUserPwd, int* piPort);
		//Load the Cfg
		//iRet = LoadCfg("cfg.txt", strIP, &iChannelIdx, strUserId, strUserPwd, &iPort);
		//if(iRet != 0)
		//{
		//	printf("load cfg error!\n");
		//	break;
		//}

		//check all the parameter
		printf("device_id:%s, start_time:%s, prefix:%s, device_ip:%s, device_idx:%d, userId:%s, userPwd:%s, port:%d\n", pIPCUId, pInputStringTime, strDirPrefix, pstrIP, iChannelIdx, pstrUserId, pstrUserPwd, iPort);

		memset(&sDvrIniCfg, 0, sizeof(NET_DVR_INIT_CFG_ABILITY));
		sDvrIniCfg.enumMaxAlarmNum = INIT_CFG_NUM_2048;
		sDvrIniCfg.enumMaxLoginUsersNum = INIT_CFG_NUM_2048;
		bRet = NET_DVR_SetSDKInitCfg(NET_SDK_INIT_CFG_ABILITY, &sDvrIniCfg);
		if (bRet == FALSE)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_SetSDKInitCfg fail, error code:%d\n", dwErr);
			break;
		}

		bRet = NET_DVR_Init();
		if (bRet == FALSE)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_Init fail, error code:%d\n", dwErr);
			break;
		}

		memset(&sUserLogin, 0, sizeof(NET_DVR_USER_LOGIN_INFO ));
		memset(&sDvrDevInfo, 0, sizeof(NET_DVR_DEVICEINFO_V40));

		strcpy(sUserLogin.sDeviceAddress, pstrIP);
		sUserLogin.wPort = iPort;
		strcpy(sUserLogin.sUserName, pstrUserId);
		strcpy(sUserLogin.sPassword, pstrUserPwd);

		lUserId = NET_DVR_Login_V40(&sUserLogin, &sDvrDevInfo);
		if (lUserId == -1)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_Login_V40 fail, error code:%d\n", dwErr);
			break;
		}


		//memset(&sTimeSpan, 0, sizeof(NET_DVR_RECORD_TIME_SPAN));
		//sTimeSpan.strBeginTime.dwYear = 2018;
		//sTimeSpan.strBeginTime.dwMonth = 3;
		//sTimeSpan.strBeginTime.dwDay = 19;
		//sTimeSpan.strBeginTime.dwHour = 12;
		//sTimeSpan.strBeginTime.dwMinute = 0;
		//sTimeSpan.strBeginTime.dwSecond = 0;

		//sTimeSpan.strBeginTime.dwYear = 2018;
		//sTimeSpan.strBeginTime.dwMonth = 3;
		//sTimeSpan.strBeginTime.dwDay = 19;
		//sTimeSpan.strBeginTime.dwHour = 12;
		//sTimeSpan.strBeginTime.dwMinute = 30;
		//sTimeSpan.strBeginTime.dwSecond = 0;

		//sTimeSpan.byType = 0;
		//sTimeSpan.dwSize = sizeof(NET_DVR_TIME) * 2 + 36;

		//memset(&sTimeSpanInquiry, 0, sizeof(NET_DVR_RECORD_TIME_SPAN_INQUIRY));
		//sTimeSpanInquiry.byType = 0;
		//sTimeSpanInquiry.dwSize = 64;

		memset(&sFileCond, 0, sizeof(NET_DVR_FILECOND_V40));
		sFileCond.dwFileType = 0;
		sFileCond.lChannel = iChannelIdx;

		sFileCond.struStartTime.dwYear = sBeginTime.iYear;
		sFileCond.struStartTime.dwMonth = sBeginTime.iMonth;
		sFileCond.struStartTime.dwDay = sBeginTime.iDay;
		sFileCond.struStartTime.dwHour = sBeginTime.iHour;
		sFileCond.struStartTime.dwMinute = sBeginTime.iMin;
		sFileCond.struStartTime.dwSecond = sBeginTime.iSec;

		AddTimeOneSec(&sBeginTime,&sEndTime);

		sFileCond.struStopTime.dwYear = sEndTime.iYear;
		sFileCond.struStopTime.dwMonth = sEndTime.iMonth;
		sFileCond.struStopTime.dwDay = sEndTime.iDay;
		sFileCond.struStopTime.dwHour = sEndTime.iHour;
		sFileCond.struStopTime.dwMinute = sEndTime.iMin;
		sFileCond.struStopTime.dwSecond = sEndTime.iSec;

		lFindHandle = NET_DVR_FindFile_V40(lUserId, &sFileCond);
		if (lFindHandle == -1)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_FindFile_V40 fail, error code:%d\n", dwErr);
			break;
		}

		do 
		{
			memset(&sFindData, 0, sizeof(NET_DVR_FINDDATA_V40));
			lFind = NET_DVR_FindNextFile_V40(lFindHandle,
				&sFindData);
			switch (lFind)
			{
				case NET_DVR_FILE_SUCCESS:
				{
					printf("start time:%d-%d-%d %d:%d:%d, end time:%d-%d-%d %d:%d:%d, file name:%s, file size:%d\n", 
						sFindData.struStartTime.dwYear, sFindData.struStartTime.dwMonth, sFindData.struStartTime.dwDay, 
						sFindData.struStartTime.dwHour, sFindData.struStartTime.dwMinute, sFindData.struStartTime.dwSecond, 
						sFindData.struStopTime.dwYear, sFindData.struStopTime.dwMonth, sFindData.struStopTime.dwDay,
						sFindData.struStopTime.dwHour, sFindData.struStopTime.dwMinute, sFindData.struStopTime.dwSecond, 
						sFindData.sFileName, sFindData.dwFileSize);
					pFileDvrArray[iFileCount] = (char*)malloc(1024);
					memset(pFileDvrArray[iFileCount], 0, 1024);
					pFilelocalArray[iFileCount] = (char*)malloc(1024);
					memset(pFilelocalArray[iFileCount], 0, 1024);
					strcpy(pFileDvrArray[iFileCount], sFindData.sFileName);
					sprintf(pFilelocalArray[iFileCount], "%s%d%02d%02d%02d%02d%02d_%d%02d%02d%02d%02d%02d.mp4",
						strDirPrefix,
						sFindData.struStartTime.dwYear, sFindData.struStartTime.dwMonth, sFindData.struStartTime.dwDay,
						sFindData.struStartTime.dwHour, sFindData.struStartTime.dwMinute, sFindData.struStartTime.dwSecond,
						sFindData.struStopTime.dwYear, sFindData.struStopTime.dwMonth, sFindData.struStopTime.dwDay,
						sFindData.struStopTime.dwHour, sFindData.struStopTime.dwMinute, sFindData.struStopTime.dwSecond);
					iFileCount++;
					break;
				}
				case NET_DVR_FILE_NOFIND:
				{
					printf("Can't find the file!\n");										
					break;
				}
				case NET_DVR_ISFINDING:
				{
					if(iState != DVR_STATE_FINDING)
					{
						iState = DVR_STATE_FINDING;
						printf("Is in finding the file!\n");					
					}
					break;
				}
				case NET_DVR_NOMOREFILE:
				{
					printf("no more file!\n");
					break;
				}
			}
		} while ((lFind == NET_DVR_FILE_SUCCESS) || (lFind == NET_DVR_ISFINDING));


		if (iFileCount > 0)
		{
			for (iIndex = 0; iIndex < iFileCount; iIndex++)
			{
				printf("dvr_file:%s, local_file:%s\n", pFileDvrArray[iIndex], pFilelocalArray[iIndex]);
				lFileHandle = NET_DVR_GetFileByName(lUserId, pFileDvrArray[iIndex],
					pFilelocalArray[iIndex]);
				if (lFileHandle != -1)
				{
					iPos = 0;
					bRet = NET_DVR_PlayBackControl_V40(lFileHandle, NET_DVR_PLAYSTART, 0, 0, 0, 0);
					if (bRet == FALSE)
					{
						dwErr = NET_DVR_GetLastError();
						printf("NET_DVR_PlayBackControl_V40, error code:%d\n", dwErr);
					}
					else
					{
						iPos = NET_DVR_GetDownloadPos(lFileHandle);
						while (iPos != 100)
						{
							sleep(1);
							iPos = NET_DVR_GetDownloadPos(lFileHandle);
						}

						NET_DVR_StopGetFile(lFileHandle);
					}
				}
				else
				{
					dwErr = NET_DVR_GetLastError();
					printf("NET_DVR_GetFileByName, error code:%d\n", dwErr);
				}
			}
		}

		bRet = NET_DVR_FindClose_V30(lFindHandle);
		if (bRet == FALSE)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_FindClose_V30 fail, error code:%d\n", dwErr);
			break;
		}



		bRet = NET_DVR_Logout(lUserId);
		if (bRet == FALSE)
		{
			dwErr = NET_DVR_GetLastError();
			printf("NET_DVR_SetSDKInitCfg fail, error code:%d\n", dwErr);
			break;
		}


	} while (0);




	bRet = NET_DVR_Cleanup();
	if (bRet == FALSE)
	{
		dwErr = NET_DVR_GetLastError();
		printf("NET_DVR_SetSDKInitCfg fail, error code:%d\n", dwErr);
	}

	for(iIndex=0; iIndex<iFileCount; iIndex++)
	{
	    if(pFileDvrArray[iIndex] != NULL)
	    {
	    	free(pFileDvrArray[iIndex]);
			pFileDvrArray[iIndex] = NULL;
	    }

		if(pFilelocalArray[iIndex] != NULL)
		{
			free(pFilelocalArray[iIndex]);
			pFilelocalArray[iIndex] = NULL;
		}
	}

	return 0;
}



int ParseCmd(int iArgc, char*  pArgv[], char**  ppIPCUId, char** ppInputStringTime, char** ppPrefix, char** ppDeviceIP, int* piChannelIdx, char ** ppDvrUId, char** ppDvrUPwd, int* piPort)
{
	int  iIndex = 0;
	int     iRet = 0;
	char*   pCurOpt = NULL;
	char*   pCurValue = NULL;


	if (iArgc < 7)
	{
		printf("invalid parameter!\n");
		printf(	"-uid          strIPCUId         set IPC UId\n"
			"-user_id      strUserID	 set the device user ID\n"
			"-user_pwd     strUserPwd	 set the device user Password\n"
			"-port         strUser Login port	 set the device user login port\n"
			"-start_time   strStartTime	 set the start time for downloading\n"
			"-device_ip    Dvr IP Address	set the DVR IP Address for downloading\n"
			"-channel_index   channel index in DVR	  set the channel index in DVR\n"
			"-start_time   strStartTime	 set the start time for downloading\n"
			"-prefix output prefix           set the dir for output, should be absolute path\n"
			);
		return ERR_INVALID_PARAMETERS;
	}

	while (iIndex < iArgc)
	{
		pCurOpt = pArgv[iIndex];
		if (strcmp("-uid", pCurOpt) == 0)
		{
			*ppIPCUId = pArgv[++iIndex];
		}

		if (strcmp("-start_time", pCurOpt) == 0)
		{
			*ppInputStringTime = pArgv[++iIndex];
		}
	
		if (strcmp("-prefix", pCurOpt) == 0)
		{
			*ppPrefix = pArgv[++iIndex];
		}
	
		if (strcmp("-user_id", pCurOpt) == 0)
		{
			*ppDvrUId = pArgv[++iIndex];
		}

		if (strcmp("-user_pwd", pCurOpt) == 0)
		{
			*ppDvrUPwd = pArgv[++iIndex];
		}

		if (strcmp("-device_ip", pCurOpt) == 0)
		{
			*ppDeviceIP = pArgv[++iIndex];
		}

		if (strcmp("-channel_index", pCurOpt) == 0)
		{
			iRet = sscanf(pArgv[++iIndex], "%d", piChannelIdx);
		}

		if (strcmp("-port", pCurOpt) == 0)
		{
			iRet = sscanf(pArgv[++iIndex], "%d", piPort);
		}

		if(iRet == -1)
		{
			printf("Invalid Parameters!\n");
			return ERR_INVALID_PARAMETERS;    
		}

		iIndex++;
	}

	return 0;
}


int LoadCfg(char*  pCfgFile, char*  pstrDeviceIP, int*  piDeviceIndex, char*  pstrUserId, char*  pstrUserPwd, int* piPort)
{
	char*   pFind = NULL;
	char*   pRetLine = NULL;
	FILE*  pFile = NULL;
	int  iRet = 0;
	int  iAttributeCount = 0;
	char   strLine[2048] = { 0 };

	do 
	{
		pFile = fopen(pCfgFile, "rb");
		if(pFile == NULL)
		{
			printf("can't open the cfg file!\n");
			iRet = ERR_INVALID_PARAMETERS;
			break;
		}


		pRetLine = fgets(strLine, 2048, pFile);
		while (pRetLine != NULL)
		{
			RemoveLineSeps(strLine);
			if (strstr(strLine, "device_ip:") != NULL)
			{
				pFind = strstr(strLine, "device_ip:");
				memcpy(pstrDeviceIP, pFind + strlen("device_ip:"), strlen(strLine) - (pFind + strlen("device_ip:") - &(strLine[0])));
			}

			if (strstr(strLine, "user_id:") != NULL)
			{
				pFind = strstr(strLine, "user_id:");
				memcpy(pstrUserId, pFind + strlen("user_id:"), strlen(strLine) - (pFind + strlen("user_id:") - &(strLine[0])));
			}

			if (strstr(strLine, "user_pwd:") != NULL)
			{
				pFind = strstr(strLine, "user_pwd:");
				memcpy(pstrUserPwd, pFind + strlen("user_pwd:"), strlen(strLine) - (pFind + strlen("user_pwd:") - &(strLine[0])));
			}

			if (strstr(strLine, "port:") != NULL)
			{
				pFind = strstr(strLine, "port:");
				sscanf(pFind+strlen("port:"), "%d", piPort);
			}

			if (strstr(strLine, "channel_index:") != NULL)
			{
				pFind = strstr(strLine, "channel_index:");
				sscanf(pFind+strlen("channel_index:"), "%d", piDeviceIndex);
			}

			memset(strLine, 0, 2048);
			pRetLine = fgets(strLine, 1024, pFile);
		}
	} while (0);

	if (pFile != NULL)
	{
		fclose(pFile);
	}

	return iRet;

}

int AddTimeOneSec(S_Full_Time* pTimeOriginal, S_Full_Time* pTimeAdded)
{
	S_Full_Time  sTimeTmp = *pTimeOriginal;

	do
	{
		if(sTimeTmp.iSec < 59)
		{
			sTimeTmp.iSec++;
			break;
		}

		if(sTimeTmp.iMin < 59)
		{
			sTimeTmp.iSec = 0;
			sTimeTmp.iMin++;
			break;
		}

		if(sTimeTmp.iHour < 23)
		{
			sTimeTmp.iSec = 0;
			sTimeTmp.iMin = 0;
			sTimeTmp.iHour++;
			break;
		}
	}while(0);

	*pTimeAdded = sTimeTmp;
	return 0;
}


