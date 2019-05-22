#include "http_client.h"

int GetCameraSnapShot(char*  pCameraSnapShotURL, unsigned char**  pImageBuf, int* pImagDataMaxSize, int*  piImageSize)
{
	int iRet = 0;

	do 
	{
		iRet = HttpGet(pCameraSnapShotURL, pImageBuf, pImagDataMaxSize, piImageSize);
	} while (0);

	return iRet;
}

