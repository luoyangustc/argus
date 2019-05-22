#include "gsoap_common.h"
#include "device_account_info.h"
#include "cJSON.h"


#ifdef __linux__
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#endif


#define MAX_GUESS_CTX_COUNT  256
#define MAX_GUESS_TRY_ITEM_COUNT  4

#define IDLE_STATE_FOR_GUESS_CTX  0
#define WORKING_STATE_FOR_GUESS   1


typedef struct
{
	char     strUUID[64];
	S_Default_Account_Info   sDefaultArray[MAX_GUESS_TRY_ITEM_COUNT];
	int      iCurArraySize;
	int      iTotalCount;
	int      iCurStartIndex;
	int      iRightIndex;
	int      iWorkingState;
} S_Uid_Pwd_Try_Ctx;


S_Uid_Pwd_Try_Ctx  g_aGuessArray[MAX_GUESS_CTX_COUNT];
int                g_iGuessCount = 0;


static S_Uid_Pwd_Try_Ctx* FindGuessItemByUUID(char*  pstrUUID)
{
	S_Uid_Pwd_Try_Ctx* pRet = NULL;
	int iIndex = 0;

	do
	{
		for(iIndex=0; iIndex<g_iGuessCount; iIndex++)
		{
			if(strcmp(pstrUUID, g_aGuessArray[iIndex].strUUID) == 0)
			{
				pRet = &(g_aGuessArray[iIndex]);
				break;
			}
		}
	}while(0);

	return pRet;
}

static S_Uid_Pwd_Try_Ctx* GetEmptyCtxForGuess()
{
	S_Uid_Pwd_Try_Ctx* pRet = NULL;
	int iIndex = 0;

	do
	{
		for (iIndex = 0; iIndex < MAX_GUESS_CTX_COUNT; iIndex++)
		{
			if (g_aGuessArray[iIndex].iWorkingState == IDLE_STATE_FOR_GUESS_CTX)
			{
				pRet = &(g_aGuessArray[iIndex]);
				break;
			}
		}
	} while (0);

	return pRet;
}

static S_Uid_Pwd_Try_Ctx* GetGuessCtxByUUID(char*  pstrUUID)
{
	S_Uid_Pwd_Try_Ctx* pRet = NULL;

	pRet = FindGuessItemByUUID(pstrUUID);

	if (pRet == NULL)
	{
		pRet = GetEmptyCtxForGuess();
		memset(pRet, 0, sizeof(S_Uid_Pwd_Try_Ctx));
		strcpy(pRet->strUUID, pstrUUID);
		pRet->iRightIndex = INVALID_ACCOUNT_IDX;
		pRet->iWorkingState = WORKING_STATE_FOR_GUESS;
		pRet->iCurArraySize = MAX_GUESS_TRY_ITEM_COUNT;
		g_iGuessCount++;
	}

	return pRet;
}


static int OnvifGetUUIDFromDiscoverProbeMsg(struct wsdd__ProbeMatchType *pProbeMatch, char* pOutUUID, char*  pXAddress, char*  pUID, char*   pPWD)
{
	int iRet = 0;
	char*   pFind = NULL;
	S_UUID_Account_Info  sUUIDInfo = { 0 };
	do 
	{
		if (pProbeMatch == NULL)
		{
			break;
		}

		strcpy(pOutUUID, pProbeMatch->wsa__EndpointReference.Address);
		strcpy(sUUIDInfo.strUUID, pProbeMatch->wsa__EndpointReference.Address);
		iRet = FindDeviceAccountInfoByUUID("uuid_info.txt", &sUUIDInfo);
		if (iRet == 0)
		{
			//printf("Can't find the account info for %s\n", sUUIDInfo.strUUID);
		}
		else
		{
			strcpy(pUID, sUUIDInfo.strUID);
			strcpy(pPWD, sUUIDInfo.strPWD);
			iRet = 1;
		}

		pFind = strstr(pProbeMatch->XAddrs, " ");
		if (pFind != NULL)
		{
			memcpy(pXAddress, pProbeMatch->XAddrs, pFind - pProbeMatch->XAddrs);
		}
		else
		{
			strcpy(pXAddress, pProbeMatch->XAddrs);
		}
	} while (0);

	return iRet;
}

static int OnvifTryUidPwd(char*  pXAddress, S_Uid_Pwd_Try_Ctx*  pUidPwdCtx)
{
	int iRet = 0;
	int iTotalCount = 0;
	int  iIndex = 0;
	S_Dev_Info  sDevIns = {0};

	do
	{
		iRet = GetDefaultAccountInfo(pUidPwdCtx->sDefaultArray, MAX_GUESS_TRY_ITEM_COUNT, 
			pUidPwdCtx->iCurStartIndex, &(pUidPwdCtx->iCurArraySize), &(pUidPwdCtx->iTotalCount));

		for(iIndex=0; iIndex<pUidPwdCtx->iCurArraySize; iIndex++)
		{
			iRet = OnvifGetDeviceBasicInfo(pUidPwdCtx->sDefaultArray[iIndex].strUID, 
											pUidPwdCtx->sDefaultArray[iIndex].strPWD, pXAddress, &sDevIns);
			if(iRet == 0)
			{
				pUidPwdCtx->iRightIndex = iIndex;
				pUidPwdCtx->iWorkingState = IDLE_STATE_FOR_GUESS_CTX;
				g_iGuessCount--;
				break;
			}
		}

		if(pUidPwdCtx->iRightIndex == INVALID_ACCOUNT_IDX)
		{
			pUidPwdCtx->iCurStartIndex += pUidPwdCtx->iCurArraySize;
		}
		
	}while(0);

	return iRet;
}

static int OnvifGetDeviceVideoURLWithProfileToken(char* pUID, char* pPwd, char*  pXAddress, char*  pToken, char* pStreamURI)
{
	int iResult = 0;
	struct tt__StreamSetup              ttStreamSetup;
	struct tt__Transport                ttTransport;
	struct _trt__GetStreamUri           sReqStream;
	struct _trt__GetStreamUriResponse   sRespStream;
	struct soap *pSoap = NULL;
	char    strRtspURL[1024] = { 0 };
	char*   pFind = NULL;

	do
	{
		SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));
		memset(&sReqStream, 0x00, sizeof(sReqStream));
		memset(&sRespStream, 0x00, sizeof(sRespStream));
		memset(&ttStreamSetup, 0x00, sizeof(ttStreamSetup));
		memset(&ttTransport, 0x00, sizeof(ttTransport));
		ttStreamSetup.Stream = tt__StreamType__RTP_Unicast;
		ttStreamSetup.Transport = &ttTransport;
		ttStreamSetup.Transport->Protocol = tt__TransportProtocol__RTSP;
		ttStreamSetup.Transport->Tunnel = NULL;
		sReqStream.StreamSetup = &ttStreamSetup;
		sReqStream.ProfileToken = pToken;

		OnvifSetAuthInfo(pSoap, pUID, pPwd);
		iResult = soap_call___trt__GetStreamUri(pSoap, pXAddress, NULL, &sReqStream, &sRespStream);
		if (iResult == 0)
		{
			//Make up rtsp url
			pFind = strstr(sRespStream.MediaUri->Uri, "rtsp://");
			strcat(pStreamURI, "rtsp://");
			strcat(pStreamURI, pUID);
			strcat(pStreamURI, ":");
			strcat(pStreamURI, pPwd);
			strcat(pStreamURI, "@");
			strcat(pStreamURI, pFind + strlen("rtsp://"));
			printf("Get Rtsp URI:%s\n", pStreamURI);
		}

	} while (0);

	if (NULL != pSoap)
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}

static int OnvifGetDeviceImageSnapURLWithProfileToken(char* pUID, char* pPwd, char*  pXAddress, char*  pToken, char* pStreamURI)
{
	int iResult = 0;
	struct soap *pSoap = NULL;
	struct _trt__GetSnapshotUri         sReq;
	struct _trt__GetSnapshotUriResponse sResp;
	char*  pFind = NULL;

	do
	{
		SOAP_ASSERT(NULL != pXAddress);
		SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));

		OnvifSetAuthInfo(pSoap, pUID, pPwd);
		memset(&sReq, 0x00, sizeof(sReq));
		memset(&sResp, 0x00, sizeof(sResp));
		sReq.ProfileToken = pToken;
		iResult = soap_call___trt__GetSnapshotUri(pSoap, pXAddress, NULL, &sReq, &sResp);
		if (iResult == 0)
		{
			//Make up rtsp url
			pFind = strstr(sResp.MediaUri->Uri, "http://");
			if(pFind != NULL)
			{	
				strcat(pStreamURI, "http://");
				strcat(pStreamURI, pUID);
				strcat(pStreamURI, ":");
				strcat(pStreamURI, pPwd);
				strcat(pStreamURI, "@");
				strcat(pStreamURI, pFind + strlen("http://"));
				printf("Get Snapshot URI:%s\n", pStreamURI);
			}
		}


	} while (0);

	if (NULL != pSoap)
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}


void  GlobalInit()
{
	memset(g_aGuessArray, 0, sizeof(S_Uid_Pwd_Try_Ctx)*MAX_GUESS_CTX_COUNT);
	g_iGuessCount = 0;
}


int InitDevDiscoverInfo(S_Dev_Discover_Info*  pDevDiscover)
{
	int  iRet = 0;
	do
	{
		if(pDevDiscover == NULL)
		{
			iRet = 1;
			break;
		}

		memset(pDevDiscover->pLatestDevArray, 0, sizeof(S_Dev_Info*)*1024);
		pDevDiscover->iMaxLatestDevCount = 1024;
		pDevDiscover->iCurLatestDevCount = 0;

		pDevDiscover->pStrJson = (char *)malloc(1024*1024);
		pDevDiscover->iBufSizeForJson = 1024*1024;
		if(pDevDiscover->pStrJson == NULL)
		{
			iRet = 1;
			break;
		}
		memset(pDevDiscover->strIP, 0, 128);		

		iRet = pthread_mutex_init(&(pDevDiscover->tMutex), NULL);
		if(iRet != 0)
		{
			iRet = 1;
			break;
		}
	}while(0);

	return iRet;
}

void UnInitDevDiscoverInfo(S_Dev_Discover_Info*  pDevDiscover)
{
	int iIndex = 0;
	do
	{
		if(pDevDiscover == NULL)
		{
			break;
		}

		for(iIndex=0; iIndex<pDevDiscover->iCurLatestDevCount; iIndex++)
		{
			if(pDevDiscover->pLatestDevArray[iIndex] != NULL)
			{
				free(pDevDiscover->pLatestDevArray[iIndex]);
				pDevDiscover->pLatestDevArray[iIndex] = NULL;
			}
		}

		if(pDevDiscover->pStrJson != NULL)
		{
			free(pDevDiscover->pStrJson);
			pDevDiscover->pStrJson = NULL;	
		}

		pthread_mutex_destroy(&(pDevDiscover->tMutex));
	}while(0);
}

int BuildDevDiscoverJSON(S_Dev_Discover_Info* pDevDiscover, char*   pFileDumpPath)
{
	int iRet = 0;
	cJSON*              pRoot = NULL;
	cJSON*              pJsonArray = NULL;
	cJSON*              pItem = NULL;
	int                 iDevCount = 0;
	int                 iIndex = 0;
	char*               pJsonString = NULL;
	FILE*               pFileDump = NULL;
	cJSON*              pProfileArray = NULL;
	cJSON*              pProfileItem = NULL;
	int                 iJsonMaxSize = 0;
	int                 iIndexProfile = 0;
	pthread_mutex_lock(&(pDevDiscover->tMutex));
	
	do
	{
		iJsonMaxSize = pDevDiscover->iBufSizeForJson; 
		iDevCount = pDevDiscover->iCurLatestDevCount;
		pRoot = cJSON_CreateObject();
		pJsonArray = cJSON_CreateArray();
		if(pRoot == NULL || pJsonArray == NULL)
		{
			iRet = 1;
			break;
		}
		
		cJSON_AddNumberToObject(pRoot, "count", iDevCount); 
		cJSON_AddItemToObject(pRoot, "Devs", pJsonArray);
		for (iIndex = 0; iIndex < iDevCount; iIndex++)
		{
			pItem = cJSON_CreateObject();
			if(pItem != NULL)
			{
				cJSON_AddNumberToObject(pItem, "index", iIndex);
				cJSON_AddStringToObject(pItem, "Manufacturer", pDevDiscover->pLatestDevArray[iIndex]->strManufacturer);
				cJSON_AddStringToObject(pItem, "Model", pDevDiscover->pLatestDevArray[iIndex]->strModel);
				cJSON_AddStringToObject(pItem, "FirmwareVersion", pDevDiscover->pLatestDevArray[iIndex]->strFirmwareVersion);
				cJSON_AddStringToObject(pItem, "SerialNumber", pDevDiscover->pLatestDevArray[iIndex]->strSerialNumber);
				cJSON_AddStringToObject(pItem, "HardwareId", pDevDiscover->pLatestDevArray[iIndex]->strHardwareId);
				cJSON_AddNumberToObject(pItem, "MediaProfile Count", pDevDiscover->pLatestDevArray[iIndex]->iMediaProfileCount);
				pProfileArray = cJSON_CreateArray();
				for (iIndexProfile = 0; iIndexProfile < pDevDiscover->pLatestDevArray[iIndex]->iMediaProfileCount; iIndexProfile++)
				{
					pProfileItem = cJSON_CreateObject();
					cJSON_AddNumberToObject(pProfileItem, "index", iIndexProfile);
					cJSON_AddNumberToObject(pProfileItem, "Width", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].iWidth);
					cJSON_AddNumberToObject(pProfileItem, "Height", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].iHeight);
					cJSON_AddNumberToObject(pProfileItem, "Bitrate(kb)", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].iBitrate);
					cJSON_AddStringToObject(pProfileItem, "Video Codec:", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].strCodec);
					cJSON_AddStringToObject(pProfileItem, "rtsp url:", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].strMainStreamRtspURL);
					cJSON_AddStringToObject(pProfileItem, "Snapshot url:", pDevDiscover->pLatestDevArray[iIndex]->sMediaProfiles[iIndexProfile].strMainStreamSnapshotURL);
					cJSON_AddItemToArray(pProfileArray, pProfileItem);
				}
				cJSON_AddItemToObject(pItem, "Media Profiles", pProfileArray);

				//cJSON_AddStringToObject(pItem, "URL", pDevDiscover->pLatestDevArray[iIndex]->strMainStreamRtspURL);
				cJSON_AddItemToArray(pJsonArray, pItem);
			}
		}

		pJsonString = cJSON_Print(pRoot);
		if(pJsonString != NULL)
		{
			memset(pDevDiscover->pStrJson, 0, iJsonMaxSize);
			strcat(pDevDiscover->pStrJson, pJsonString);
			printf("json:\n%s\n", pJsonString);
			free(pJsonString);
			pFileDump = fopen(pFileDumpPath, "wb");
			if(pFileDump != NULL)
			{
				fwrite(pDevDiscover->pStrJson, 1, strlen(pDevDiscover->pStrJson), pFileDump);
				fclose(pFileDump);
			}
		}

	}while(0);

	pthread_mutex_unlock(&(pDevDiscover->tMutex));
	
	if(iRet == 0)
	{
		cJSON_Delete(pRoot);
	}
	else	
	{
		cJSON_Delete(pRoot);
		cJSON_Delete(pJsonArray);
	}

	return 0;
}


void SoapPerror(struct soap *pSoap, const char *pStr)
{
	do 
	{
		if (pSoap == NULL || pStr == NULL)
		{
			break;
		}

		if (NULL == pStr)
		{
			SOAP_DBGERR("[soap] error: %d, %s, %s\n", pSoap->error, *soap_faultcode(pSoap), *soap_faultstring(pSoap));
		}
		else
		{
			SOAP_DBGERR("[soap] %s error: %d, %s, %s\n", pStr, pSoap->error, *soap_faultcode(pSoap), *soap_faultstring(pSoap));
		}
	} while (0);

	return;
}

struct soap *OnvifSoapNew(int iSocketTimeOut, char*  pStrIp)
{
	struct soap *pSoap = NULL;                                                 
	struct in_addr if_req;

	do 
	{
		pSoap = soap_new();
		if (pSoap == NULL)
		{
			break;
		}

		// set soap namespaces
		soap_set_namespaces(pSoap, namespaces);

		// set timeout for socket
		pSoap->recv_timeout = iSocketTimeOut;                                    
		pSoap->send_timeout = iSocketTimeOut;
		pSoap->connect_timeout = iSocketTimeOut;

#if defined(__linux__) || defined(__linux)                                          
		pSoap->socket_flags = MSG_NOSIGNAL;                                          
#endif
		// set UTF-8 encoding
		soap_set_mode(pSoap, SOAP_C_UTFSTRING);

		if (pStrIp != NULL && strlen(pStrIp) != 0)
		{
			if_req.s_addr = inet_addr(pStrIp);  // binding IP address
			pSoap->ipv4_multicast_if = (char*)soap_malloc(pSoap, sizeof(struct in_addr));
			memset(pSoap->ipv4_multicast_if, 0, sizeof(struct in_addr));
			memcpy(pSoap->ipv4_multicast_if, (char*)&if_req, sizeof(if_req));
		}
	} while (0);

	return pSoap;
}


void OnvifSoapDelete(struct soap *pSoap)
{
	if (pSoap != NULL)
	{
		// remove deserialized class instances (C++ only)
		soap_destroy(pSoap); 

		// Clean up deserialized data (except class instances) and temporary data
		soap_end(pSoap);
		
		// Reset, close communications, and remove callbacks
		soap_done(pSoap);
		
		// Reset and deallocate the context created with soap_new or soap_copy
		soap_free(pSoap);                                                            
	}
}

void* ONVIFSoapMalloc(struct soap *pSoap, unsigned int iBuffSize)
{
	void *p = NULL;

	if (iBuffSize > 0)
	{
		p = soap_malloc(pSoap, iBuffSize);
		SOAP_ASSERT(NULL != p);
		memset(p, 0x00, iBuffSize);
	}
	return p;
}

int OnvifSetAuthInfo(struct soap *pSoap, const char *pUserName, const char *pPassword)
{
	int result = 0;
	SOAP_ASSERT(NULL != pUserName);
	SOAP_ASSERT(NULL != pPassword);

	result = soap_wsse_add_UsernameTokenDigest(pSoap, NULL, pUserName, pPassword);

	return result;
}


void OnvifInitHeader(struct soap *pSoap)
{
	struct SOAP_ENV__Header *pheader = NULL;

	SOAP_ASSERT(NULL != pSoap);

	pheader = (struct SOAP_ENV__Header *)ONVIFSoapMalloc(pSoap, sizeof(struct SOAP_ENV__Header));
	soap_default_SOAP_ENV__Header(pSoap, pheader);
	pheader->wsa__MessageID = (char*)soap_wsa_rand_uuid(pSoap);
	pheader->wsa__To = (char*)ONVIFSoapMalloc(pSoap, strlen(SOAP_TO) + 1);
	pheader->wsa__Action = (char*)ONVIFSoapMalloc(pSoap, strlen(SOAP_ACTION) + 1);
	strcpy(pheader->wsa__To, SOAP_TO);
	strcpy(pheader->wsa__Action, SOAP_ACTION);
	pSoap->header = pheader;
	return;
}

void OnvifInitProbeType(struct soap *pSoap, struct wsdd__ProbeType *pProbe)
{
	struct wsdd__ScopesType *pScope = NULL;                                      

	SOAP_ASSERT(NULL != pSoap);
	SOAP_ASSERT(NULL != pProbe);

	pScope = (struct wsdd__ScopesType *)ONVIFSoapMalloc(pSoap, sizeof(struct wsdd__ScopesType));
	
	// set the device scope
	soap_default_wsdd__ScopesType(pSoap, pScope);  
	pScope->__item = (char*)ONVIFSoapMalloc(pSoap, strlen(SOAP_ITEM) + 1);
	strcpy(pScope->__item, SOAP_ITEM);

	memset(pProbe, 0x00, sizeof(struct wsdd__ProbeType));
	soap_default_wsdd__ProbeType(pSoap, pProbe);
	pProbe->Scopes = pScope;

	// set the device type
	pProbe->Types = (char*)ONVIFSoapMalloc(pSoap, strlen(SOAP_TYPES) + 1);
	strcpy(pProbe->Types, SOAP_TYPES);
}

void OnvifDiscoverDevices(S_Dev_Discover_Info*  pDevDiscoverInfo)
{
	int iIndex = 0;
	int iResult = 0;
	unsigned int iDevCount = 0;
	struct soap *pSoap = NULL;
	struct wsdd__ProbeType      sReq;
	struct __wsdd__ProbeMatches sResp;
	struct wsdd__ProbeMatchType *pProbeMatch = NULL;
	char         strXAddr[1024] = { 0 };
	char         strUID[1024] = { 0 };
	char         strPWD[1024] = { 0 };
	char         strUUID[1024] = { 0 };
	int          iRetForGetDevBasic = 0;
	int          iRetForGetDevVideo = 0;
	int          iRetForGetDevCap = 0;
	int          iRetForGetProfiles = 0;
	int          iRetForGetImagesSnap = 0;
	int          iRetFindPwd = 0;
	S_Uid_Pwd_Try_Ctx*     pGuessCtx = NULL;
	S_Dev_Info    sDevIns;
	S_UUID_Account_Info   sUUIDAccount;


	SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, pDevDiscoverInfo->strIP)));

	do 
	{
		//Set the Header
		OnvifInitHeader(pSoap);

		//Set the device and scope
		OnvifInitProbeType(pSoap, &sReq);

		//Send the Probe Message MultiCast
		iResult = soap_send___wsdd__Probe(pSoap, SOAP_MCAST_ADDR, NULL, &sReq);
		while (SOAP_OK == iResult)
		{
			memset(&sResp, 0x00, sizeof(sResp));
			iResult = soap_recv___wsdd__ProbeMatches(pSoap, &sResp);
			if (SOAP_OK == iResult) 
			{
				if (pSoap->error) 
				{
					SoapPerror(pSoap, "ProbeMatches");
				}
				else 
				{
					if (NULL != sResp.wsdd__ProbeMatches)
					{
						sResp.wsdd__ProbeMatches->__sizeProbeMatch;
						for (iIndex = 0; iIndex< sResp.wsdd__ProbeMatches->__sizeProbeMatch; iIndex++)
						{
							memset(strUUID, 0, 1024);
							memset(strXAddr, 0, 1024);
							memset(strUID, 0, 1024);
							memset(strPWD, 0, 1024);
							
							memset(&sDevIns, 0, sizeof(S_Dev_Info));

							pProbeMatch = sResp.wsdd__ProbeMatches->ProbeMatch + iIndex;
							iRetFindPwd = OnvifGetUUIDFromDiscoverProbeMsg(pProbeMatch, strUUID, strXAddr, strUID, strPWD);
							if(iRetFindPwd == 0)
							{
								
								pGuessCtx = GetGuessCtxByUUID(strUUID);
								if(pGuessCtx == NULL)
								{
									continue;
								}

								GetDefaultAccountInfo(pGuessCtx->sDefaultArray, 
										MAX_GUESS_TRY_ITEM_COUNT, pGuessCtx->iCurStartIndex, 
										&(pGuessCtx->iCurArraySize), &(pGuessCtx->iTotalCount));
								OnvifTryUidPwd(strXAddr, pGuessCtx);
								if(pGuessCtx->iRightIndex != INVALID_ACCOUNT_IDX)
								{
									strcpy(strUID, pGuessCtx->sDefaultArray[pGuessCtx->iRightIndex].strUID);
									strcpy(strPWD, pGuessCtx->sDefaultArray[pGuessCtx->iRightIndex].strPWD);
									AddDeviceUUIDAccountInfo("uuid_info.txt", strUUID, strUID, strPWD);
								}
								else
								{
									if(pGuessCtx->iCurStartIndex == pGuessCtx->iTotalCount)
									{
										pGuessCtx->iCurStartIndex = 0;
									}
									continue;
								}

							}
							
							iRetForGetDevBasic = OnvifGetDeviceBasicInfo(strUID, strPWD, strXAddr, &sDevIns);
							iRetForGetDevCap = OnvifGetCapabilities(strUID, strPWD, strXAddr, &sDevIns);
							iRetForGetProfiles = OnvifGetProfiles(strUID, strPWD, sDevIns.strMediaAddr, &sDevIns);
							iRetForGetDevVideo = OnvifGetDeviceVideo(strUID, strPWD, sDevIns.strMediaAddr, &sDevIns);
							iRetForGetImagesSnap = OnvifGetDeviceImage(strUID, strPWD, sDevIns.strMediaAddr, &sDevIns);
							if (iRetForGetDevBasic == 0 && iRetForGetDevVideo == 0 && iDevCount < pDevDiscoverInfo->iMaxLatestDevCount)
							{
								pthread_mutex_lock(&(pDevDiscoverInfo->tMutex));
								if (pDevDiscoverInfo->pLatestDevArray[iDevCount] == NULL)
								{
									pDevDiscoverInfo->pLatestDevArray[iDevCount] = (S_Dev_Info* )malloc(sizeof(S_Dev_Info));
									if (pDevDiscoverInfo->pLatestDevArray[iDevCount] == NULL)
									{
										continue;
									}
									else
									{
										memset(pDevDiscoverInfo->pLatestDevArray[iDevCount], 0, sizeof(S_Dev_Info));
										memcpy(pDevDiscoverInfo->pLatestDevArray[iDevCount], &sDevIns, sizeof(S_Dev_Info));
									}
								}
								else
								{
								
									memset(pDevDiscoverInfo->pLatestDevArray[iDevCount], 0, sizeof(S_Dev_Info));
									memcpy(pDevDiscoverInfo->pLatestDevArray[iDevCount], &sDevIns, sizeof(S_Dev_Info));
								}

								iDevCount++;
								pDevDiscoverInfo->iCurLatestDevCount = iDevCount;
								pthread_mutex_unlock(&(pDevDiscoverInfo->tMutex));
							}
						}
					}
				}
			}
			else if (pSoap->error) 
			{
				printf("Error:%s\n", soap_faultstring(pSoap));
				break;
			}
		}

		if (NULL != pSoap)
		{
			OnvifSoapDelete(pSoap);
		}

		BuildDevDiscoverJSON(pDevDiscoverInfo, "Devs.json");
	} while (0);

	return;
}


int OnvifGetDeviceBasicInfo(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo)
{
	int iResult = 0;
	struct soap *pSoap = NULL;
	struct _tds__GetDeviceInformation           devinfo_req;
	struct _tds__GetDeviceInformationResponse   devinfo_resp;

	SOAP_ASSERT(NULL != pXAddress);
	SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));
	do
	{
		OnvifSetAuthInfo(pSoap, pUID, pPwd);
		memset(&devinfo_req, 0x00, sizeof(devinfo_req));
		memset(&devinfo_resp, 0x00, sizeof(devinfo_resp));

		iResult = soap_call___tds__GetDeviceInformation(pSoap, pXAddress, NULL, &devinfo_req, &devinfo_resp);
		if (iResult == 0)
		{
			strcpy(pDevInfo->strManufacturer, devinfo_resp.Manufacturer);
			strcpy(pDevInfo->strModel, devinfo_resp.Model);
			strcpy(pDevInfo->strFirmwareVersion, devinfo_resp.FirmwareVersion);
			strcpy(pDevInfo->strSerialNumber, devinfo_resp.SerialNumber);
			strcpy(pDevInfo->strHardwareId, devinfo_resp.HardwareId);
		}

	} while (0);


	if (NULL != pSoap)
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}


int OnvifGetCapabilities(char* pUID, char* pPwd, char *pXAddress, S_Dev_Info*  pDevInfo)
{
	int iResult = 0;
	struct soap *pSoap = NULL;
	struct _tds__GetCapabilities            sReq;
	struct _tds__GetCapabilitiesResponse    sResp;

	do 
	{
		SOAP_ASSERT(NULL != pXAddress);
		SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));

		OnvifSetAuthInfo(pSoap, pUID, pPwd);

		memset(&sReq, 0x00, sizeof(sReq));
		memset(&sResp, 0x00, sizeof(sResp));
		iResult = soap_call___tds__GetCapabilities(pSoap, pXAddress, NULL, &sReq, &sResp);
		if (iResult == 0)
		{
			if (sResp.Capabilities->Analytics != NULL && sResp.Capabilities->Analytics->XAddr != NULL)
			{
				strcat(pDevInfo->strAnalyticsAddr, sResp.Capabilities->Analytics->XAddr);
			}

			if (sResp.Capabilities->Device != NULL && sResp.Capabilities->Device->XAddr != NULL)
			{
				strcat(pDevInfo->strDeviceAddr, sResp.Capabilities->Device->XAddr);
			}

			if (sResp.Capabilities->Events != NULL && sResp.Capabilities->Events->XAddr != NULL)
			{
				strcat(pDevInfo->strEventsAddr, sResp.Capabilities->Events->XAddr);
			}

			if (sResp.Capabilities->Imaging != NULL && sResp.Capabilities->Imaging->XAddr != NULL)
			{
				strcat(pDevInfo->strImagingAddr, sResp.Capabilities->Imaging->XAddr);
			}

			if (sResp.Capabilities->Media != NULL && sResp.Capabilities->Media->XAddr != NULL)
			{
				strcat(pDevInfo->strMediaAddr, sResp.Capabilities->Media->XAddr);
			}

			if (sResp.Capabilities->PTZ != NULL && sResp.Capabilities->PTZ->XAddr != NULL)
			{
				strcat(pDevInfo->strPTZAddr, sResp.Capabilities->PTZ->XAddr);
			}
		}

	} while (0);

	if (NULL != pSoap) 
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}

int OnvifGetProfiles(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo)
{
	int iIndex = 0;
	int iResult = 0;
	char*   pToken = NULL;
	struct soap *pSoap = NULL;
	struct _trt__GetProfiles sReq;
	struct _trt__GetProfilesResponse    sResp;
	do
	{
		SOAP_ASSERT(NULL != pXAddress);
		SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));

		OnvifSetAuthInfo(pSoap, pUID, pPwd);

		memset(&sReq, 0x00, sizeof(sReq));
		memset(&sResp, 0x00, sizeof(sResp));
		iResult = soap_call___trt__GetProfiles(pSoap, pXAddress, NULL, &sReq, &sResp);
		if (iResult == 0)
		{
			//Get the all Profiles
			for (iIndex=0; iIndex<sResp.__sizeProfiles && iIndex<MAX_MEDIA_PROFILE_COUNT; iIndex++)
			{
				if (sResp.Profiles[iIndex].VideoEncoderConfiguration != NULL)
				{
					pDevInfo->sMediaProfiles[iIndex].iHeight = sResp.Profiles[iIndex].VideoEncoderConfiguration->Resolution->Height;
					pDevInfo->sMediaProfiles[iIndex].iWidth = sResp.Profiles[iIndex].VideoEncoderConfiguration->Resolution->Width;
					switch (sResp.Profiles[iIndex].VideoEncoderConfiguration->Encoding)
					{
					case tt__VideoEncoding__H264:
					{
						strcpy(pDevInfo->sMediaProfiles[iIndex].strCodec, "H264");
						break;
					}
					case tt__VideoEncoding__MPEG4:
					{
						strcpy(pDevInfo->sMediaProfiles[iIndex].strCodec, "MPEG4");
						break;
					}
					case tt__VideoEncoding__JPEG:
					{
						strcpy(pDevInfo->sMediaProfiles[iIndex].strCodec, "MJPEG");
						break;
					}
					}

					strcpy((char *)(pDevInfo->sMediaProfiles[iIndex].strMediaProfile), sResp.Profiles[iIndex].token);
					pDevInfo->sMediaProfiles[iIndex].iBitrate = sResp.Profiles[iIndex].VideoEncoderConfiguration->RateControl->BitrateLimit;
				}
			}

			pDevInfo->iMediaProfileCount = iIndex;
		}
	} while (0);

	if (NULL != pSoap)
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}


int OnvifGetDeviceVideo(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo)
{
	int iResult = 0;
	int iIndex = 0;
	char    strRtspURL[1024] = { 0 };

	do 
	{
		for (iIndex = 0; iIndex < pDevInfo->iMediaProfileCount; iIndex++)
		{
			memset(strRtspURL, 0, 1024);
			iResult = OnvifGetDeviceVideoURLWithProfileToken(pUID, pPwd, pXAddress, pDevInfo->sMediaProfiles[iIndex].strMediaProfile, strRtspURL);
			if (iResult == 0)
			{
				strcpy(pDevInfo->sMediaProfiles[iIndex].strMainStreamRtspURL, strRtspURL);
			}
		}
	} while (0);

	return 0;
}

int OnvifGetDeviceImage(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo)
{
	int iResult = 0;
	int iIndex = 0;
	char    strRtspURL[1024] = { 0 };

	do
	{
		for (iIndex = 0; iIndex < pDevInfo->iMediaProfileCount; iIndex++)
		{
			memset(strRtspURL, 0, 1024);
			iResult = OnvifGetDeviceImageSnapURLWithProfileToken(pUID, pPwd, pXAddress, pDevInfo->sMediaProfiles[iIndex].strMediaProfile, strRtspURL);
			if (iResult == 0)
			{
				strcpy(pDevInfo->sMediaProfiles[iIndex].strMainStreamSnapshotURL, strRtspURL);
			}
		}
	} while (0);

	return 0;
}

