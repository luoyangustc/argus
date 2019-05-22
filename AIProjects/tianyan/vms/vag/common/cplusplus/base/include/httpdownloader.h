#ifndef __HTTP_DOWNLOADER_H__
#define __HTTP_DOWNLOADER_H__

#ifdef _WINDOWS
#include <WinSock2.h>
#else
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "typedef_win.h"

#endif//_WINDOWS


#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
using namespace boost;

class CHttpDownloader
{
public:
    CHttpDownloader(void);
    ~CHttpDownloader(void);

    bool TestFileExist(const string& sURL);
    bool Post(const string& sURL);
    bool DownloadFiles(const string& sURL,unsigned int& out_len,shared_array<char>& out_response);
	int GetStatusCode()const{return m_iStatus;}
private:
    bool parseURL();
    bool GetSevrSockAddr(struct sockaddr_in& addr);
    int  GetBaseInfo(const string& sURL,const char *szMethod);
    void Init();
    bool DownloadFiles_Part(char * out_buff,unsigned int llStart,unsigned int llEnd);
#ifdef _WINDOWS
	void MBS2EncodeURL(const string& sURL,OUT string& sEncodeURL);
#endif//_WINDOWS
private:
	char m_szBuff[1024];
    char m_szLastURL[4096];
    string m_sURL;
    string m_sEncodeURL;
    string m_sGet;
    string m_sHost;
    string m_sHostname;

    unsigned short m_wPort;
    unsigned int m_dwIP;

	string m_sRelativePath;
private:
    struct sockaddr_in m_saddr_serv;
    int m_iStatus;
    unsigned int m_llFileLength;
#ifdef _WINDOWS
    SOCKET m_sockfd;
#else
	int m_sockfd;
#endif

};

#define VD_RECV_BUFF_LEN (16*1024)

#endif//__HTTP_DOWNLOADER_H__

