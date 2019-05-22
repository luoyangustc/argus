#ifndef __GSOAP_COMMON_H__
#define __GSOAP_COMMON_H__


#include <assert.h>

#include "soapH.h"
#include "wsaapi.h"
#include "wsseapi.h"

#define SOAP_ASSERT     assert
#define SOAP_DBGLOG     printf
#define SOAP_DBGERR     printf
#define SOAP_TO         "urn:schemas-xmlsoap-org:ws:2005:04:discovery"
#define SOAP_ACTION     "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe"

//onvif multicast addr
#define SOAP_MCAST_ADDR "soap.udp://239.255.255.250:3702"  

//devcie scope
#define SOAP_ITEM       "onvif://www.onvif.org"                                 

//device type
#define SOAP_TYPES      "dn:NetworkVideoTransmitter"     

//socket timeout value (second)
#define SOAP_SOCK_TIMEOUT    (10)                                               
#define MAX_SIZE_FOR_UUID    128
#define MAX_SIZE_FOR_IP		 128
#define MAX_SIZE_FOR_ONVIF_DESC    1024
#define MAX_MEDIA_PROFILE_COUNT    8


typedef struct  
{
	int iWidth;
	int iHeight;
	int iBitrate;
	char   strMediaProfile[64];
	char   strCodec[64];
	char strMainStreamRtspURL[MAX_SIZE_FOR_ONVIF_DESC];
	char strMainStreamSnapshotURL[MAX_SIZE_FOR_ONVIF_DESC];
}S_Media_Profile;

typedef struct
{
	char strManufacturer[MAX_SIZE_FOR_ONVIF_DESC];
	char strModel[MAX_SIZE_FOR_ONVIF_DESC];
	char strFirmwareVersion[MAX_SIZE_FOR_ONVIF_DESC];
	char strSerialNumber[MAX_SIZE_FOR_ONVIF_DESC];
	char strHardwareId[MAX_SIZE_FOR_ONVIF_DESC];

	char  strAnalyticsAddr[MAX_SIZE_FOR_ONVIF_DESC];
	char  strDeviceAddr[MAX_SIZE_FOR_ONVIF_DESC];
	char  strEventsAddr[MAX_SIZE_FOR_ONVIF_DESC];
	char  strImagingAddr[MAX_SIZE_FOR_ONVIF_DESC];
	char  strMediaAddr[MAX_SIZE_FOR_ONVIF_DESC];
	char  strPTZAddr[MAX_SIZE_FOR_ONVIF_DESC];

	unsigned int    ulEventCapabilityValue;
	unsigned int    ulAnalyticsCapabilityValue;



	int  iMediaProfileCount;
	S_Media_Profile   sMediaProfiles[MAX_MEDIA_PROFILE_COUNT];
} S_Dev_Info;


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void SoapPerror(struct soap *pSoap, const char *pStr);
struct soap *OnvifSoapNew(int iSocketTimeOut, char*  pStrIp);
void OnvifSoapDelete(struct soap *pSoap);
void* OnvifSoapMalloc(struct soap *pSoap, unsigned int iBuffSize);

int OnvifSetAuthInfo(struct soap *pSoap, const char *pUserName, const char *pPassword);
void OnvifInitHeader(struct soap *pSoap);
void OnvifInitProbeType(struct soap *pSoap, struct wsdd__ProbeType *pProbe);

int OnvifGetDeviceBasicInfo(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo);
int OnvifGetCapabilities(char* pUID, char* pPwd, char *pXAddress, S_Dev_Info*  pDevInfo);

int OnvifGetDeviceVideo(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo);
int OnvifGetDeviceImage(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo);
int OnvifGetProfiles(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo);
int OnvifGetEventCapabilities(char* pUID, char* pPwd, char*  pXAddress, S_Dev_Info*  pDevInfo);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
