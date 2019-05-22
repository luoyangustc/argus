#include "gsoap_common.h"

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
			if (pFind != NULL)
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
				pDevInfo->ulAnalyticsCapabilityValue |= (sResp.Capabilities->Analytics->RuleSupport);
				pDevInfo->ulAnalyticsCapabilityValue |= (sResp.Capabilities->Analytics->AnalyticsModuleSupport << 1);
			}

			if (sResp.Capabilities->Device != NULL && sResp.Capabilities->Device->XAddr != NULL)
			{
				strcat(pDevInfo->strDeviceAddr, sResp.Capabilities->Device->XAddr);
			}

			if (sResp.Capabilities->Events != NULL && sResp.Capabilities->Events->XAddr != NULL)
			{
				strcat(pDevInfo->strEventsAddr, sResp.Capabilities->Events->XAddr);
				pDevInfo->ulEventCapabilityValue |= (sResp.Capabilities->Events->WSSubscriptionPolicySupport);
				pDevInfo->ulEventCapabilityValue |= ((sResp.Capabilities->Events->WSPullPointSupport) << 1);
				pDevInfo->ulEventCapabilityValue |= ((sResp.Capabilities->Events->WSPausableSubscriptionManagerInterfaceSupport) << 2);
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
		if (iResult != -1)
		{
			//Get the all Profiles
			pToken = sResp.Profiles->token;
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

int OnvifGetEventCapabilities(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo)
{
	int iResult = 0;
	struct soap *pSoap = NULL;
	struct _tev__GetEventProperties         sReq;
	struct _tev__GetEventPropertiesResponse sResp;

	do
	{
		SOAP_ASSERT(NULL != pXAddress);
		SOAP_ASSERT(NULL != (pSoap = OnvifSoapNew(SOAP_SOCK_TIMEOUT, NULL)));

		OnvifSetAuthInfo(pSoap, pUID, pPwd);
		memset(&sReq, 0x00, sizeof(sReq));
		memset(&sResp, 0x00, sizeof(sResp));
		iResult = soap_call___tev__GetEventProperties(pSoap, pXAddress, NULL, &sReq, &sResp);
		if (iResult != 0)
		{
			if (pSoap->error)
			{
				printf("Error:%s\n", soap_faultstring(pSoap));
				break;
			}
		}
	} while (0);

	if (NULL != pSoap)
	{
		OnvifSoapDelete(pSoap);
	}

	return iResult;
}
