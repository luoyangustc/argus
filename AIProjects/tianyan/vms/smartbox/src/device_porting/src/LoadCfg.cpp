#include "LoadCfg.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "common_def.h"


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


int LoadCfg(char*  pstrCfgFilePath, char* pstrSmartBoxBZBaseURL, char* pstrSmartBoxBZSettingURL, char*  pstrSmartBoxBZChannelURL, char* pstrSmartBoxfDeviceID, char*  pstrEntryServerIp, int*  piEntryServerPort)
{
	int iRet = 0;
	FILE*   pFile = NULL;
	char    strLine[1024] = { 0 };
	char*   pLine = NULL;
	char*   pFind = NULL;

	do 
	{
		if (pstrCfgFilePath == NULL || strlen(pstrCfgFilePath) == 0)
		{
			iRet = ERR_INVALID_PARAMETER;
			break;
		}

		pFile = fopen(pstrCfgFilePath, "rb");
		if (pFile == NULL)
		{
			iRet = ERR_OPEN_URL_FAIL;
			break;
		}

		while (!feof(pFile))
		{
			memset(strLine, 0, 1024);
			pLine = fgets(strLine, 1024, pFile);
			if (pLine == NULL)
			{
				continue;
			}

			RemoveLineSeps(strLine);
			if (strstr(strLine, "SmartBoxBZSetting:"))
			{
				pFind = strstr(strLine, "SmartBoxBZSetting:");
				strcpy(pstrSmartBoxBZSettingURL, pFind + strlen("SmartBoxBZSetting:"));
			}
			else if (strstr(strLine, "SmartBoxBZBase:"))
			{
				pFind = strstr(strLine, "SmartBoxBZBase:");
				strcpy(pstrSmartBoxBZBaseURL, pFind + strlen("SmartBoxBZBase:"));
			}
			else if (strstr(strLine, "SmartBoxChannelInfo:"))
			{
				pFind = strstr(strLine, "SmartBoxChannelInfo:");
				strcpy(pstrSmartBoxBZChannelURL, pFind + strlen("SmartBoxChannelInfo:"));
			}
			else if (strstr(strLine, "DeviceId:"))
			{
				pFind = strstr(strLine, "DeviceId:");
				strcpy(pstrSmartBoxfDeviceID, pFind + strlen("DeviceId:"));
			}
			else if (strstr(strLine, "Entry_Server_Ip:"))
			{
				pFind = strstr(strLine, "Entry_Server_Ip:");
				strcpy(pstrEntryServerIp, pFind + strlen("Entry_Server_Ip:"));
			}
			else if (strstr(strLine, "Entry_Server_Port:"))
			{
				pFind = strstr(strLine, "Entry_Server_Port:");
				*piEntryServerPort = atoi(pFind + strlen("Entry_Server_Port:"));
			}
		}
	} while (0);

	return iRet;
}