#ifndef __GET_DEVICEINFO_FROM_BZ_SETTING_H__
#define __GET_DEVICEINFO_FROM_BZ_SETTING_H__

#include "SmartBox_porting.h"


int GetDeviceInfoFromBZSetting(char* pBZBaseInterface, char* pBZInterface, char*  pStrSmarboxDeviceId, S_Channel_Map_Info*  pChannelNodeArray, int iArrayMaxSize, int*  pChannelNodeCount, int*  pMaxChannelNodeCount);
int GetChannelInfoFromBZSetting(char* pBZInterface, S_Channel_Map_Info*  pChannelNode, int iIndex);

#endif __GET_DEVICEINFO_FROM_BZ_SETTING_H__