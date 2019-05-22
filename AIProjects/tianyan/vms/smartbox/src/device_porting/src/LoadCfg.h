#ifndef __LOAD_CFG_H__
#define __LOAD_CFG_H__

//Make sure pSmartBoxBZSettingULR   and  pSmartBoxfDeviceID have enough memory space 
int LoadCfg(char*  pstrCfgFilePath, char* pstrSmartBoxBZBaseURL, char* pstrSmartBoxBZSettingURL, char* pstrSmartBoxChannelURL, char* pstrSmartBoxfDeviceID, char*  pstrEntryServerIp, int*  piEntryServerPort);

#endif
