#ifndef _AY_NET_CORE_CFG_
#define _AY_NET_CORE_CFG_

#include <string>
#include "typedefine.h"
using namespace std;

class CAYNetCoreCfg
{
public:
	CAYNetCoreCfg();
	~CAYNetCoreCfg();
    
    int ReadCfg(const string& cfg_file);
    string GetCfgFile(){return m_strCfgFileName;}
    uint32 GetClientIoServiceNum(){return m_unClientIoServiceNum;}
    uint32 GetClientIoServiceWorks(){return m_unClientIoServiceWorks;}
    uint32 GetClientTaskServiceNum(){return m_unClientTaskServiceNum;}
    uint32 GetClientTaskServiceWorks(){return m_unClientTaskServiceWorks;}
    uint32 GetServerIoServiceNum(){return m_unServerIoServiceNum;}
    uint32 GetServerIoServiceWorks(){return m_unServerIoServiceWorks;}
    uint32 GetServerTaskServiceNum(){return m_unServerTaskServiceNum;}
    uint32 GetServerTaskServiceWorks(){return m_unServerTaskServiceWorks;}
public:
    int GetLogLevel(){return m_unLogLevel;}
    void SetLogLevel(int level){m_unLogLevel = level;}

    int GetLogSize(){return m_unLogMaxSize;}
    void SetLogSize(uint32 max_size){m_unLogMaxSize = max_size;}
private:
    string m_strCfgFileName;
    uint32 m_unClientIoServiceNum;
    uint32 m_unClientIoServiceWorks;
    uint32 m_unClientTaskServiceNum;
    uint32 m_unClientTaskServiceWorks;
    uint32 m_unServerIoServiceNum;
    uint32 m_unServerIoServiceWorks;
    uint32 m_unServerTaskServiceNum;
    uint32 m_unServerTaskServiceWorks;
    uint32 m_unLogLevel;
    uint32 m_unLogMaxSize;
};

#endif