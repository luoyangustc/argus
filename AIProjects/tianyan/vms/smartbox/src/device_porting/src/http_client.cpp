#include <curl/curl.h>
#include "http_client.h"
#include "common_def.h"
#include "string.h"
#include "stdlib.h"

typedef struct
{
	unsigned char*   pBuffer;
	int iCurMaxSize;
	int iCurDataSize;
} S_Write_Handle;

static size_t write_handle(char * data, size_t size, size_t nmemb, void* p)
{
	int iRetSize = 0;
	unsigned char*  pBufNew = NULL;
	S_Write_Handle*   pHandle = (S_Write_Handle*)p;


	if (pHandle->iCurMaxSize <= (size*nmemb + pHandle->iCurDataSize))
	{
		pHandle->iCurMaxSize = pHandle->iCurMaxSize * 2 + size*nmemb;
		pBufNew = (unsigned char*)realloc(pHandle->pBuffer, pHandle->iCurMaxSize);
		if (pBufNew == NULL)
		{
			pHandle->iCurMaxSize = 0;
			pHandle->iCurDataSize = 0;
			return 0;
		}
		else
		{
			pHandle->pBuffer = pBufNew;
		}
	}
	
	memcpy(pHandle->pBuffer + pHandle->iCurDataSize, data, size*nmemb);
	pHandle->iCurDataSize += (size*nmemb);
	iRetSize = size*nmemb;
	return iRetSize;
}

int  InitWriteHandle(S_Write_Handle*  psWriteHandle)
{
	int iRet = 0;
	do
	{
		psWriteHandle->pBuffer = (unsigned char*)malloc(512 * 1024);
		BREAK_IN_NULL_POINTER(psWriteHandle->pBuffer, iRet, ERR_LACK_MEMORY);

		psWriteHandle->iCurMaxSize = 512 * 1024;
		memset(psWriteHandle->pBuffer, 0, 512 * 1024);
	} while (0);

	return iRet;
}

int  UnInitWriteHandle(S_Write_Handle*  psWriteHandle)
{
	if (psWriteHandle != NULL)
	{
		SAFE_FREE(psWriteHandle->pBuffer);
	}
	return 0;
}

int ConvertHttpURLForAuth(char* pInputURL, char*  pOutputURL, char* pUserName, char*  pPassWord)
{
	int iRet = -1;
	char*  pFindStart = NULL;
	char*  pFindEnd = NULL;
	char*  pFindSep = NULL;
	char*  pStrEnd = NULL;
	char*  pFindFirstSep = NULL;

	do
	{
		//Find ://
		pFindStart = strstr(pInputURL, "://");
		if (pFindStart == NULL)
		{
			break;
		}

		pStrEnd = pInputURL + strlen(pInputURL);

		pFindStart = pFindStart + strlen("://");
		pFindFirstSep = strstr(pFindStart, "/");
		pFindEnd = strstr(pFindStart, "@");
		if (pFindEnd == NULL)
		{
			//Normal URL
			strcpy(pOutputURL, pInputURL);
			iRet = 0;
			break;
		}
		else
		{
			if (pFindFirstSep < pFindEnd)
			{
				//Normal URL
				strcpy(pOutputURL, pInputURL);
				iRet = 0;
				break;
			}
		}


		pFindSep = strstr(pFindStart, ":");
		if (pFindEnd == NULL)
		{
			break;
		}

		memcpy(pOutputURL, pInputURL, pFindStart - pInputURL);
		memcpy(pOutputURL + (pFindStart - pInputURL), pFindEnd + 1, pStrEnd - pFindEnd - 1);
		memcpy(pUserName, pFindStart, pFindSep - pFindStart);
		memcpy(pPassWord, pFindSep + 1, pFindEnd - pFindSep - 1);

		iRet = 0;

	} while (0);

	return iRet;
}


int HttpGet(char*  pHttpGetURL, unsigned char**  ppOutput, int* piOutputMaxSize, int*  piOutputSize)
{
	S_Write_Handle     sWriterH;
	int                iRet = 1;
	CURLcode ret;
	CURL *hnd = NULL;
	int  iRespCode = 0;
	struct curl_slist *slist1 = NULL;
	FILE*   pFileImage = NULL;
	char    strOutput[1024] = { 0 };
	char    strUserName[128] = { 0 };
	char    strUserPwd[128] = { 0 };
	unsigned char*   pBufNew = NULL;

	do
	{
		BREAK_IN_NULL_POINTER(pHttpGetURL, iRet, ERR_INVALID_PARAMETER);
		BREAK_IN_ERR_ZERO(strlen(pHttpGetURL), iRet, ERR_INVALID_PARAMETER);
		memset(&sWriterH, 0, sizeof(S_Write_Handle));
		InitWriteHandle(&sWriterH);

		iRet = ConvertHttpURLForAuth(pHttpGetURL, strOutput, strUserName, strUserPwd);
		if (iRet != 0)
		{
			break;
		}

		hnd = curl_easy_init();
		BREAK_IN_NULL_POINTER(hnd, iRet, ERR_CURL_INNNER);

		ret = curl_easy_setopt(hnd, CURLOPT_URL, strOutput);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		ret = curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		ret = curl_easy_setopt(hnd, CURLOPT_CONNECTTIMEOUT, 1);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		ret = curl_easy_setopt(hnd, CURLOPT_TIMEOUT, 1);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		if (strlen(strUserPwd) != 0)
		{
			curl_easy_setopt(hnd, CURLOPT_HTTPAUTH, (long)CURLAUTH_ANY);
			curl_easy_setopt(hnd, CURLOPT_PASSWORD, strUserPwd);
			curl_easy_setopt(hnd, CURLOPT_USERNAME, strUserName);
			curl_easy_setopt(hnd, CURLOPT_HTTPAUTH, CURLAUTH_BASIC | CURLAUTH_DIGEST);
		}

		curl_easy_setopt(hnd, CURLOPT_TIMEOUT, 2L);
		/* Define our callback to get called when there's data to be written */
		ret = curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, write_handle);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		/* Set a pointer to our struct to pass to the callback */
		ret = curl_easy_setopt(hnd, CURLOPT_WRITEDATA, &sWriterH);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);


		//ret = curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);
		//BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		ret = curl_easy_perform(hnd);
		//BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		ret = curl_easy_getinfo(hnd, CURLINFO_RESPONSE_CODE, &iRespCode);
		BREAK_IN_ERR_NOT_ZERO(ret, iRet, ERR_CURL_INNNER);

		if (ret == 0)
		{
			if (iRespCode == 200)
			{
				if (sWriterH.iCurDataSize > 0)
				{
					if (*piOutputMaxSize < sWriterH.iCurDataSize)
					{
						pBufNew = (unsigned char*)realloc(*ppOutput, sWriterH.iCurDataSize+1);
						if (pBufNew == NULL)
						{
							iRet = ERR_LACK_MEMORY;
							break;
						}

						*piOutputMaxSize = sWriterH.iCurDataSize + 1;
						*ppOutput = pBufNew;
					}
					
					memcpy(*ppOutput, sWriterH.pBuffer, sWriterH.iCurDataSize);
					*piOutputSize = sWriterH.iCurDataSize;
					iRet = 0;
				}
				else
				{
					iRet = ERR_CURL_INNNER;
				}
			}
			else if (iRespCode == 401)
			{
				printf("Meet 401\n");
			}
		}

	} while (0);


	if (hnd != NULL)
	{
		curl_easy_cleanup(hnd);
	}


	printf("Resource Size:%d\n", *piOutputSize);
	//pFileImage = fopen("test.jpg", "wb");
	//if (pOutput != NULL)
	//{
	//	fwrite(pOutput, 1, *piOutputSize, pFileImage);
	//	fclose(pFileImage);
	//}

	SAFE_FREE(sWriterH.pBuffer);
	return (int)iRet;
}
