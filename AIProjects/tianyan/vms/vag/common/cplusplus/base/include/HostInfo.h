#ifndef __HOSTINFO_H__
#define __HOSTINFO_H__

#ifndef _WINDOWS
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <string.h>
#else
#include <WinSock2.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif //

#include "typedef_win.h"
#include "typedefine.h"
//#include "./CriticalSectionMgr.h"
#include <string>
#include <list>
using namespace std;

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
using namespace boost;

#pragma pack(1)

class CHostInfo
{
public:
	CHostInfo()	
	{
		IP = inet_addr("0.0.0.0");
		Port = 0;
	}
	CHostInfo(string strHost,USHORT nPort)
	{
        IP = inet_addr(strHost.c_str());
        if( IP == 0xffffffff ) //INADDR_NONE
        {
            hostent * hostx;
            hostx = ::gethostbyname(strHost.c_str());
            if(hostx)
                IP = *((u_long*)hostx->h_addr_list[0]);
        }
        Port = nPort;
	}
	CHostInfo(DWORD dwIP,USHORT nPort)
	{
		IP = dwIP;
		Port = nPort;
	}

	CHostInfo(const CHostInfo & hi)
	{
		IP = hi.IP;
		Port = hi.Port;
	}

	void SetNodeString(string strIP, string strPort, bool bUseDNS = false)
	{
		strIP += ":";
		strIP += strPort;
		SetNodeString(strIP.c_str(),bUseDNS);
	}

	void GetNodeString(string& strIP, string& strPort) const
	{
		strIP = inet_ntoa(*(in_addr*)&IP);
		//char buff[16];
		strPort = boost::lexical_cast<string>((unsigned int)(Port));
	}

	void SetNodeString(const char * szNode,bool bUseDNS = false)
    {
        IP = 0;
        Port = 0;
        if(szNode)
        {
            string strNode(szNode);
            int iNext = static_cast<int>(strNode.find(':'));
            string host;
            if(iNext>0)
            {
                host = strNode.substr(0,iNext);
                IP = inet_addr(strNode.substr(0,iNext).c_str());
                iNext +=1;
                Port = atoi(strNode.substr(iNext,strNode.size()-iNext).c_str());
            }
            else
                host = strNode;
            IP = inet_addr(host.c_str());
            if((IP == 0xFFFFFFFF ||IP == INADDR_NONE) && bUseDNS)
            {
                hostent * hostx;
                hostx = ::gethostbyname(host.c_str());
                if(hostx)
                    IP = *((u_long*)hostx->h_addr_list[0]);
            }
        }
    }
    
	string GetNodeString() const
    {
        char szBuf[30];
        sprintf(szBuf,"%s:%d",inet_ntoa(*(in_addr*)&IP),Port);	
        return szBuf;
    }

	void Clear(){IP = 0;Port = 0;}
    BOOL IsEmpty()const{return IP==0&&Port==0;}
	BOOL IsValid()const
    {
        unsigned char * pIP = (unsigned char*)&IP;
        if(pIP[0] == 0 || pIP[0] == 255 || Port == 0)
        {
           return FALSE; 
        }
        return TRUE;
    }
	BOOL IsPrivate()const
    {
        unsigned char * pIP = (unsigned char*)&IP;
        if(pIP[0] == 127 ||	pIP[0] == 10 
        || (pIP[0] == 172 && (pIP[1]>=16 && pIP[1]<32)) 
        || (pIP[0] == 192 && pIP[1] == 168))
        return TRUE;
        return FALSE;
    }
    
    void GetIP(string& strIP) const
    {
        strIP = inet_ntoa(*(in_addr*)&IP);
    }

	DWORD GetIP()const
	{
		return IP;
	}

	DWORD GetPort()const
	{
		return Port;
	}
	bool operator != (const CHostInfo & first) const{return memcmp(this,&first,sizeof(CHostInfo))!=0?true:false;}
	bool operator == (const CHostInfo & first) const{return memcmp(this,&first,sizeof(CHostInfo))==0?true:false;}
	bool operator < (const CHostInfo & first) const{return memcmp(this,&first,sizeof(CHostInfo))<0?true:false;}

#ifndef _WINDOWS
	DWORD		IP;
	WORD		Port;
#else
	uint32		IP;
	uint16		Port;
#endif
};

#pragma pack()

#endif
