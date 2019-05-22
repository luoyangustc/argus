#include "device_account_info.h"
#include "stdio.h"
#include "string.h"



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

int  AddDeviceUUIDAccountInfo(char*  pInfoURL, char*  pStrUUID, char*  pStrUid, char* pStrPwd)
{
	//
	FILE*   pFile = NULL;
	char    strDump[4096] = { 0 };
	pFile = fopen(pInfoURL, "ab");
	
	if (pFile != NULL)
	{
		sprintf(strDump, "%s,%s,%s\n", pStrUUID, pStrUid, pStrPwd);
		fwrite(strDump, 1, strlen(strDump), pFile);
		fclose(pFile);
	}

	return 0;
}

int  FindDeviceAccountInfoByUUID(char*   pInfoURL, S_UUID_Account_Info*  pUUIDInfo)
{
	int iRet = 0;
	S_UUID_Account_Info   sUUIDIns;
	FILE*   pFile = NULL;
	char    strline[1024] = { 0 };
	pFile = fopen(pInfoURL, "rb");
	char*  pItem = NULL;

	do 
	{
		if (pFile == NULL)
		{
			break;
		}

		while (!feof(pFile))
		{
			memset(strline, 0, 1024);
			memset(&sUUIDIns, 0, sizeof(S_UUID_Account_Info));
			fgets(strline, 1024, pFile);
			RemoveLineSeps(strline);

			pItem = strtok(strline, ",");
			if(pItem != NULL)
			{
				strcpy(sUUIDIns.strUUID, pItem);
			}

			pItem = strtok(NULL, ",");
			if(pItem != NULL)
			{
				strcpy(sUUIDIns.strUID, pItem);
			}

			pItem = strtok(NULL, ",");
			if(pItem != NULL)
			{
				strcpy(sUUIDIns.strPWD, pItem);
			}

			if (strcmp(sUUIDIns.strUUID, pUUIDInfo->strUUID) == 0)
			{
				strcpy(pUUIDInfo->strUID, sUUIDIns.strUID);
				strcpy(pUUIDInfo->strPWD, sUUIDIns.strPWD);
				iRet = 1;
				break;
			}
		}
	} while (0);

	if (pFile != NULL)
	{
		fclose(pFile);
	}

	return iRet;
}

int  GetDefaultAccountInfo(S_Default_Account_Info*  pAccountInfo, int iAccountArraySize, int iStartIndex, int* piCount, int* piTotalCount)
{
	int iRet = 0;
	FILE*   pFile = NULL;
	char  strLine[1024] = {0};
	int   iIndex = 0;
	int   iEndIndex = 0;
	int   iTryCount = 5;
	char*  pItem = NULL;
	int   iIndexInner = 0;
	int   iNeedCount = 0;

	do
	{
		pFile = fopen("default_UidPwd.txt", "rb");
		if(pFile == NULL)
		{
			break;
		}

		iNeedCount = *piCount;
		if(iNeedCount > 0 && iNeedCount < 10)
		{
			iNeedCount = iTryCount;
		}

		iEndIndex = iStartIndex + iNeedCount;
		while (!feof(pFile))
		{
			memset(strLine, 0, 1024);
			fgets(strLine, 1024, pFile);
			RemoveLineSeps(strLine);
			if(iStartIndex <= iIndex && iIndex < iEndIndex)
			{
				if(iIndexInner < iAccountArraySize)
				{
					pItem = strtok(strLine, ",");
					if(pItem == NULL)
					{
						continue;
					}

					strcpy(pAccountInfo[iIndexInner].strUID, pItem);

					pItem = strtok(NULL, ",");
					if(pItem == NULL)
					{
						continue;
					}
					strcpy(pAccountInfo[iIndexInner].strPWD, pItem);
					iIndexInner++;
				}
			}

			iIndex++;
		}
	}while(0);

	if(pFile != NULL)
	{
		fclose(pFile);
	}

	*piCount = iIndexInner++;
	*piTotalCount = iIndex;
	return iIndexInner;
}
