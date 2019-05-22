#ifndef __DEVICE_ACCOUNT_INFO__
#define __DEVICE_ACCOUNT_INFO__

typedef struct
{
	char    strUID[64];
	char    strPWD[64];
} S_Default_Account_Info;

typedef struct
{
	char     strUUID[64];
	char     strUID[64];
	char     strPWD[64];
} S_UUID_Account_Info;

#define INVALID_ACCOUNT_IDX 0xfff

int  AddDeviceUUIDAccountInfo(char*  pInfoURL, char*  pStrUUID, char*  pStrUid, char* pStrPwd);
int  FindDeviceAccountInfoByUUID(char*   pInfoURL, S_UUID_Account_Info*  pUUIDInfo);
int  GetDefaultAccountInfo(S_Default_Account_Info*  pAccountInfo, int iAccountArraySize, int iStartIndex, int* piCount, int* piTotalCount);

#endif
