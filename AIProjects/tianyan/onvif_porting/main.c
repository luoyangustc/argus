#include "gsoap_common.h"
#include "httpd_inner.h"

void*   DevDiscoverThreadFunc(void* pArg);



int main(int argc, char **argv)
{
	unsigned int   ulPort = 0;
	char*    pIP = NULL;
	int iRet = 0;
	S_Dev_Discover_Info     sDevDiscoverInfo;
	pthread_t   t_Discover;
	void*     pThreadResult = NULL;	

	do
	{
		sscanf(argv[2], "%d", &ulPort);		
		pIP = argv[1];

		iRet = InitDevDiscoverInfo(&sDevDiscoverInfo);
		if(iRet != 0)
		{
			printf("fatal error, can't create dev discover context!\n");
			break;
		}

		strcpy(sDevDiscoverInfo.strIP, pIP);
		sDevDiscoverInfo.iThreadRunFlag = 1;
		iRet = pthread_create(&t_Discover, NULL, DevDiscoverThreadFunc, &sDevDiscoverInfo);  
		if(iRet != 0)
		{
			perror("pthread_create failed\n");
			exit(EXIT_FAILURE);
		}
		
		ServiceRun((unsigned short)ulPort, &(sDevDiscoverInfo));
	}while(0);
	
	sDevDiscoverInfo.iThreadRunFlag = 0;
	iRet = pthread_join(t_Discover, &pThreadResult);
	if (iRet != 0)
	{
		perror("pthread_join failed\n");
	}

	UnInitDevDiscoverInfo(&sDevDiscoverInfo);
	return 0;
}


void*  DevDiscoverThreadFunc(void* pArg)
{
	int iRet = 0;
	S_Dev_Discover_Info*     pDevDiscoverInfo = (S_Dev_Discover_Info *)pArg;
	while(1)
	{
		OnvifDiscoverDevices(pDevDiscoverInfo);
		usleep(10000);
	}
	return NULL;
}
