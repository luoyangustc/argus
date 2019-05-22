#include "MediaSimulator.h"
#include "CameraSnapShot.h"
#include "GetCameraInfoFromOnvif.h"
#include "stdio.h"
#include "qiniu_dev_net_porting.h"
#include "stdlib.h"
#include "string.h"



int ParseCmd(int iArgc, char*  pArgv[], char**  ppOnvifInterface, char**  ppLoacalDeviceInfo, int* piDevcieIndex, int* piProfileIndex);

int main(int argc, char* argv[])
{
	unsigned char*   pImageBuf =(unsigned char*) malloc(1024 * 1024);
	int              iImageSize = 0;
	char*             pstrOnvifInterfaceURL = NULL;
	char*             pstrLocalDeviceInfoURL = NULL;
	int iIndex = 0;
	int iRet = 0;
	S_SDK_Ins    sSdkIns;
	S_Media_Simulator    sMediaSims[8];
	S_Media_Simulator*    psMediaSims = NULL;

	S_Camera_Info        asCameraInfo[8];
	int                  iCamCount = 0;
	S_Frame              sFrameArray[8];
	int                  iProfileIndex = 0;
	int                  iDeviceIndex = 0;

	int iDeviceCount = 0;

	do 
	{

		memset(asCameraInfo, 0, sizeof(S_Camera_Info)*8);
		ParseCmd(argc, argv, &pstrOnvifInterfaceURL, &pstrLocalDeviceInfoURL, &iDeviceIndex, &iProfileIndex);
		
		iRet = GetCamerasInfoFromOnvif(pstrOnvifInterfaceURL, asCameraInfo, 8, &iCamCount);

		psMediaSims = &(sMediaSims[0]);
		
		InitSDK(&sSdkIns, &(asCameraInfo[iDeviceIndex]));
		InitMediaSimulator(psMediaSims);


		printf("Demo DeviceID:%s\n", GetDeviceId(&sSdkIns));
		

		SetRtspStreamURL(&sSdkIns, asCameraInfo[iDeviceIndex].aMediaProfiles[iProfileIndex].strRtspURL);
		SetImageSnapShotURL(&sSdkIns, asCameraInfo[iDeviceIndex].aMediaProfiles[iProfileIndex].strSnapShotURL);

		iRet = OpenMedia(psMediaSims, asCameraInfo[iDeviceIndex].aMediaProfiles[iProfileIndex].strRtspURL);
		if (iRet != 0)
		{
			break;
		}

		sFrameArray[0].pFrameData = sSdkIns.pFrameBuffer;
		iRet = ReadFrame(psMediaSims, &(sFrameArray[iIndex]));
		while (iRet == 0)
		{
			SdkStreamInfoReport(&sSdkIns, &(sFrameArray[iIndex]));
			iRet = ReadFrame(psMediaSims, &(sFrameArray[iIndex]));
		}

	} while (0);


	getchar();


	return 0;
}



int ParseCmd(int iArgc, char*  pArgv[], char**  ppOnvifInterface, char**  ppLoacalDeviceInfo, int* piDevcieIndex, int* piProfileIndex)
{
	int  iIndex = 0;
	float    fRate = 0;
	int     iRet = 0;
	char*   pCurOpt = NULL;
	char*   pCurValue = NULL;

	*piDevcieIndex = -1;
	*piProfileIndex = -1;

	if (iArgc < 2)
	{
		printf("invalid parameter!\n");
		printf("-onvif device_onvif_discover_url    set the onvif discover url\n"
			"-local device_local_url        set local device infor url(json file)\n"
			"-dev_idx device index          set the device index \n"
			"-profile_idx media profile index    set the media profile index \n"
			);
		return ERR_INVALID_PARAMETERS;
	}

	while (iIndex < iArgc)
	{
		pCurOpt = pArgv[iIndex];
		if (strcmp("-onvif", pCurOpt) == 0)
		{
			*ppOnvifInterface = pArgv[++iIndex];
		}

		if (strcmp("-local", pCurOpt) == 0)
		{
			*ppLoacalDeviceInfo = pArgv[++iIndex];
		}

		if (strcmp("-dev_idx", pCurOpt) == 0)
		{
			iRet = sscanf(pArgv[++iIndex], "%d", piDevcieIndex);
		}

		if (strcmp("-profile_idx", pCurOpt) == 0)
		{
			iRet = sscanf(pArgv[++iIndex], "%d", piProfileIndex);
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