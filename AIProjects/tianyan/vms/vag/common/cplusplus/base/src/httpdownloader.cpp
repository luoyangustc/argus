#include "httpdownloader.h"
#include "vdcharsetutil.h"
#include <boost/shared_array.hpp>
#include <boost/format.hpp>
using namespace boost;

#undef UNICODE

#ifdef _WINDOWS
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>

// link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
#else
#endif//_WINDOWS

CHttpDownloader::CHttpDownloader(void)
{
	memset(m_szLastURL, 0, sizeof(m_szLastURL));
}

CHttpDownloader::~CHttpDownloader(void)
{
}

void CHttpDownloader::Init()
{
	memset(m_szLastURL, 0, sizeof(m_szLastURL));
	m_iStatus = -1;
	m_llFileLength = 0;
	m_sockfd = -1;
	memset( & m_saddr_serv,0,sizeof( m_saddr_serv) );

	m_sURL.clear();
	m_sEncodeURL.clear();
	m_sGet.clear();
	m_sHost.clear();
	m_sHostname.clear();

	m_wPort = 0 ;
	m_dwIP = 0;
}

bool CHttpDownloader::DownloadFiles(const string& sURL,unsigned int& out_len,shared_array<char>& out_response)
{
	do 
	{
		{
			bool is_get_base_info = false;
			int iCode = 0;
			string sURLTmp = sURL;
			do
			{
				if( sURLTmp.empty() )
				{
					break;
				}
				iCode = GetBaseInfo(sURLTmp,"HEAD");
				if( iCode >= 200 && iCode < 300)
				{
					if( m_llFileLength > 0 )
					{
						is_get_base_info = true;
					}
					break;
				}

				if( iCode >= 301 && iCode <= 303 )
				{
					sURLTmp = m_szLastURL;
					continue;
				}
				break;
			}while(true);

			if( !is_get_base_info )
			{
				break;
			}
		}

		do 
		{
			if( m_llFileLength <= 0 )
			{
				break;
			}

			out_response = shared_array<char>(new char[m_llFileLength+1]);
			(out_response.get())[m_llFileLength] = 0;

			if( DownloadFiles_Part(out_response.get(),0,m_llFileLength-1) )
			{
				out_len = m_llFileLength;
#ifdef _WINDOWS
				closesocket(m_sockfd);
#else
					close(m_sockfd);
#endif//_WINDOWS
				m_sockfd = -1;

				return true;
			}
		} while (false);

#ifdef _WINDOWS
				closesocket(m_sockfd);
#else
				close(m_sockfd);
#endif//_WINDOWS
		m_sockfd = -1;			  	
	} while (false);   

	return false;
}

bool CHttpDownloader::DownloadFiles_Part(char * out_buff,unsigned int llStart,unsigned int llEnd)
{
#ifdef _WINDOWS
	SOCKET sockfd = socket(AF_INET,SOCK_STREAM,0);
	if( sockfd < 0 )
	{
		perror("create socket faild");
		return false;
	}

	int recvTime = 3000;
	int optionlen = sizeof(recvTime);
	setsockopt(sockfd,SOL_SOCKET,SO_RCVTIMEO,(char * )&recvTime,optionlen);

	recvTime = 3000;
	optionlen = sizeof(recvTime);
	setsockopt(sockfd,SOL_SOCKET,SO_SNDTIMEO,(char * )&recvTime,optionlen);
#else
	int sockfd = socket(AF_INET,SOCK_STREAM,0);
	if( sockfd < 0 )
	{
		perror("create socket faild");
		return false;
	}

	struct timeval recvTime;
	recvTime.tv_sec = 5;
	recvTime.tv_usec = 0;
	socklen_t optionlen = sizeof(recvTime);
	setsockopt(sockfd,SOL_SOCKET,SO_RCVTIMEO,(char * )&recvTime,optionlen);
#endif//_WINDOWS
	

	if( connect( sockfd,(struct sockaddr*)&m_saddr_serv,sizeof(m_saddr_serv)) < 0)
	{
		perror("connect faild");
#ifdef _WINDOWS
		closesocket(sockfd);
#else
		close(sockfd);
#endif
		return false;
	}

	char szRequest[4096];
	sprintf_s(szRequest,sizeof(szRequest),"GET %s HTTP/1.1\r\n"
		"Accept: */*\r\n"
		"Accept-Language: en-us\r\n"
		//"Accept-Encoding: gzip,deflate\r\n"
		//"User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.0; MyIE2; Alexa Toolbar)\r\n"
		"User-Agent: p2p\r\n"
		"Host: %s:%d\r\n"
		"Range:bytes=%u-%u\r\n"
		"Connection: Keep-Alive\r\n"
		"\r\n",
		m_sGet.c_str(),
		m_sHostname.c_str(),
		m_wPort,
		llStart,llEnd);

	size_t iRequestLen = strlen(szRequest);
	if( send(sockfd,szRequest,(int)iRequestLen,0) != iRequestLen )
	{
		perror("send faild");
#ifdef _WINDOWS
		closesocket(sockfd);
#else
		close(sockfd);
#endif
		return false;
	}

	bool bGetHead = false;
	int iStatus = -1;
	while(true)
	{
		char szBuff[1024];	
		unsigned int recv_want_len = 1024;

		if( bGetHead )
		{
			if( llStart > llEnd )
			{
				break;
			}

			recv_want_len = llEnd-llStart + 1;
			if( recv_want_len > 1024 )
			{
				recv_want_len = 1024;
			}
		}

		int recvbytes = recv(sockfd,szBuff,recv_want_len,0);//MSG_WAITALL

		if( recvbytes < 0 )
		{
			//perror("recv faild");
#ifdef _WINDOWS
		closesocket(sockfd);
#else
		close(sockfd);
#endif
			return false;
		}

		if( recvbytes == 0 )
		{
			if( bGetHead )
			{
				if( llStart > llEnd )
				{
					break;
				}
			}

			perror("remote shutdowned");
#ifdef _WINDOWS
		closesocket(sockfd);
#else
		close(sockfd);
#endif
			return false;
		}

		if( bGetHead )
		{
			memcpy(out_buff + llStart,szBuff,recvbytes);
			llStart += recvbytes;

			if( llStart > llEnd )
			{
				break;
			}
		}
		else
		{
			if( iStatus == -1 )
			{
				char * pStatusPos = strstr(szBuff," ");
				if( pStatusPos != NULL )
				{
					++pStatusPos;
					iStatus = atoi(pStatusPos);

					if( iStatus != 206 && iStatus != 200 )
					{
						perror("status != 206");
#ifdef _WINDOWS
						closesocket(sockfd);
#else
						close(sockfd);
#endif
						sockfd = -1;
						//s_download_file_stat.Log_Debug(LLOG_LEVEL_DEBUG,"%s:%d:%s,faild!%d,%u",__FUNCTION__,__LINE__,m_sURL.c_str(),iStatus,m_llFileLength) ;

						return false;
					}
				}
			}

			char * pEnd = strstr(szBuff,"\r\n\r\n");
			if( pEnd != NULL )
			{
				pEnd += 4;

				bGetHead = true;

				int iHeadLen = int(pEnd - szBuff);

				int iDataLen = recvbytes - iHeadLen;

				memcpy(out_buff + llStart,pEnd,iDataLen);
				llStart += iDataLen;
			}	
		}

		//printf("%s\n",szBuff);
	}

#ifdef _WINDOWS
		closesocket(sockfd);
#else
		close(sockfd);
#endif

	return true;    
}

bool CHttpDownloader::GetSevrSockAddr(struct sockaddr_in& addr)
{    
	memset( & addr,0,sizeof( addr) );
	addr.sin_family = AF_INET;
	addr.sin_port = htons(m_wPort);

	addr.sin_addr.s_addr = inet_addr(m_sHostname.c_str());
#ifdef _WINDOWS
	if( addr.sin_addr.s_addr == INADDR_NONE )
	{
		bool ret_code = false;
		{
			struct addrinfo *answer, hint;
			memset(&hint,0,sizeof(hint));
			hint.ai_family = AF_INET;
			hint.ai_socktype = SOCK_STREAM;
			hint.ai_protocol = IPPROTO_TCP;

			if( 0 == getaddrinfo(m_sHostname.c_str(), NULL, &hint, &answer) )
			{
				for (struct addrinfo * curr = answer; curr != NULL; curr = curr->ai_next) {
					if (curr->ai_family == AF_INET )
					{
						struct sockaddr_in * addr_tmp = (struct sockaddr_in *)curr->ai_addr;
						addr.sin_addr.S_un.S_addr = addr_tmp->sin_addr.S_un.S_addr;
					}

				}	   
				freeaddrinfo(answer);
				return true;
			}
			else
			{
				return false;
			}   	
		}
	}
#else
	if( addr.sin_addr.s_addr == INADDR_NONE )
	{
#ifdef __APPLE__
		{
			struct addrinfo *answer, hint;
			memset(&hint,0,sizeof(hint));
			hint.ai_family = AF_INET;
			hint.ai_socktype = SOCK_STREAM;
			hint.ai_protocol = IPPROTO_TCP;
            
			if( 0 == getaddrinfo(m_sHostname.c_str(), NULL, &hint, &answer) )
			{
				for (struct addrinfo * curr = answer; curr != NULL; curr = curr->ai_next) {
					if (curr->ai_family == AF_INET )
					{
						struct sockaddr_in * addr_tmp = (struct sockaddr_in *)curr->ai_addr;
						addr.sin_addr.s_addr = addr_tmp->sin_addr.s_addr;
					}
                    
				}
				freeaddrinfo(answer);
				return true;
			}
			else
			{
				return false;
			}   	
		}
#else
		struct hostent ret,*phost;
		memset(&ret,0,sizeof(ret));
		char dns_buff[8192];
		int rc;

		if( 0 == gethostbyname_r(m_sHostname.c_str(),&ret,dns_buff,8192,&phost,&rc)
				&& rc == 0)
		{
			addr.sin_addr.s_addr = *(unsigned int*)(ret.h_addr);
		}
		else
		{
			return false;
		}
#endif
	}
#endif//_WINDOWS

	return true;
}

bool CHttpDownloader::Post(const string& sURL)
{
	int iStatus = GetBaseInfo(sURL,"POST");
	if( iStatus >= 200 && iStatus < 300 )
	{
		return true;
	}    

	return false;
}

int CHttpDownloader::GetBaseInfo(const string& sURL,const char *szMethod)
{
	Init();
	m_sURL = sURL;
	if( ! parseURL() )
	{
		return m_iStatus;
	}

	if( ! GetSevrSockAddr(m_saddr_serv) )
	{
		return m_iStatus;
	}


	m_sockfd = socket(AF_INET,SOCK_STREAM,0);
	if( m_sockfd < 0 )
	{
		perror("create socket faild");
		return m_iStatus;
	}
#ifdef _WINDOWS
	int recvTime = 3000;
	int optionlen = sizeof(recvTime);
	setsockopt(m_sockfd,SOL_SOCKET,SO_RCVTIMEO,(char * )&recvTime,optionlen);

	recvTime = 3000;
	optionlen = sizeof(recvTime);
	setsockopt(m_sockfd,SOL_SOCKET,SO_SNDTIMEO,(char * )&recvTime,optionlen);
#else
	struct timeval recvTime;
	recvTime.tv_sec = 5;
	recvTime.tv_usec = 0;
	socklen_t optionlen = sizeof(recvTime);
	setsockopt(m_sockfd,SOL_SOCKET,SO_RCVTIMEO,(char * )&recvTime,optionlen);
	setsockopt(m_sockfd,SOL_SOCKET,SO_SNDTIMEO,(char * )&recvTime,optionlen);
#endif//_WINDOWS
	if( connect( m_sockfd,(struct sockaddr*)&m_saddr_serv,sizeof(m_saddr_serv)) < 0)
	{
		perror("connect faild");
#ifdef _WINDOWS
		closesocket(m_sockfd);
#else
		close(m_sockfd);
#endif
		m_sockfd = -1;
		return m_iStatus;
	}

	char szRequest[4096];
	sprintf_s(szRequest,sizeof(szRequest),"%s %s HTTP/1.1\r\n"
		"Accept: */*\r\n"
		"Accept-Language: en-us\r\n"
		//"Accept-Encoding: gzip,deflate\r\n"
		//"User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.0; MyIE2; Alexa Toolbar)\r\n"
		"User-Agent: p2p\r\n"
		"Host: %s:%d\r\n"
		"Connection: Keep-Alive\r\n"
		"\r\n",
		szMethod,
		m_sGet.c_str(),
		m_sHostname.c_str(),
		m_wPort);

	//printf("%s",szRequest);

	size_t iRequestLen = strlen(szRequest);
	if( send(m_sockfd,szRequest,(int)iRequestLen,0) != iRequestLen )
	{
		perror("send faild");
#ifdef _WINDOWS
		closesocket(m_sockfd);
#else
		close(m_sockfd);
#endif
		m_sockfd = -1;
		return m_iStatus;
	}

	while(true)
	{
		char szBuff[1024];	
		int recvbytes = recv(m_sockfd,szBuff,1024,0);

		if( recvbytes < 0 )
		{
			perror("recv faild");
#ifdef _WINDOWS
		closesocket(m_sockfd);
#else
		close(m_sockfd);
#endif
			m_sockfd = -1;
			return m_iStatus;
		}

		if( recvbytes == 0 )
		{
			perror("remote shutdowned");

			if( m_iStatus <200 || m_iStatus >=300 )
			{
#ifdef _WINDOWS
		closesocket(m_sockfd);
#else
		close(m_sockfd);
#endif
				m_sockfd = -1;
				return m_iStatus;
			}
			break;
		}


		//printf("%d\n%s\n",recvbytes,szBuff);
		/*
		for( int i = 0; i< recvbytes; ++i )
		{
		unsigned int a = szBuff[i];

		printf("%2x ",a);
		}

		printf("123456\n");
		*/	
		if( m_iStatus == -1 )
		{
			char * pStatusPos = strstr(szBuff," ");
			if( pStatusPos != NULL )
			{
				++pStatusPos;
				m_iStatus = atoi(pStatusPos);

				if( m_iStatus >= 301 && m_iStatus <= 303 )
				{

					char * pLocationPos = strstr(szBuff,"Location:");
					if( pLocationPos != NULL )
					{
						pLocationPos = strchr( pLocationPos,':' );
						pLocationPos ++;
						char * pLocationStartPos = strstr( pLocationPos,"http://" );

						char * pLocationEndPos = strstr( pLocationPos,"\r\n" );
						if( pLocationEndPos && pLocationStartPos && pLocationEndPos > pLocationStartPos )
						{
							int iLocationLen = int(pLocationEndPos-pLocationStartPos);
#ifdef _WINDOWS
							strncpy_s(m_szLastURL,sizeof(m_szLastURL),pLocationStartPos,iLocationLen);
#else
							strncpy(m_szLastURL,pLocationStartPos,iLocationLen);
#endif//_WINDOWS
							m_szLastURL[iLocationLen] = 0;
						}
						else
						{
#ifdef _WINDOWS
							closesocket(m_sockfd);
#else
							close(m_sockfd);
#endif //_WINDOWS
							m_sockfd = -1;
							m_iStatus = -1;
							return m_iStatus;
						}
					}
				}
				else if( m_iStatus < 200 || m_iStatus >= 300 )
				{
					perror("status != 200");
#ifdef _WINDOWS
					closesocket(m_sockfd);
#else
					close(m_sockfd);
#endif //_WINDOWS
					m_sockfd = -1;
					return m_iStatus;
				}
			}
		}

		if( m_llFileLength == 0 && 
			(m_iStatus >= 200&&m_iStatus<300) )
		{
			char * pLenPos = strstr(szBuff,"Length:");
			if( pLenPos != NULL )
			{
				pLenPos = strchr( pLenPos,':' );
				pLenPos ++;

#ifdef _WINDOWS
				m_llFileLength = _atoi64(pLenPos);	 
#else
				m_llFileLength = atoll(pLenPos);	
#endif//_WINDOWS

			}
		}

		{
			char * pEnd = strstr(szBuff,"\r\n\r\n");
			if( pEnd != NULL )
			{
				pEnd += 4;
				int iHeadLen = pEnd - szBuff;
				//m_iBuffLen = recvbytes - iHeadLen;

				//memcpy( m_szBuff,pEnd,m_iBuffLen );
				break;
			}
		}
	}

	{
#ifdef _WINDOWS
		closesocket(m_sockfd);
#else
		close(m_sockfd);
#endif
		m_sockfd = -1;
	}

	if ( m_iStatus == 200 )
	{
		return m_iStatus;
	}
	return m_iStatus;
}

bool CHttpDownloader::TestFileExist(const string& sURL)
{
	string sURLTmp = sURL;
	do
	{
		if( sURLTmp.empty() )
		{
			break;
		}
		int iCode = GetBaseInfo(sURLTmp,"HEAD");
		if( iCode >= 200 && iCode < 300)
		{
			if( m_llFileLength > 0 )
			{
				return true;
			}
			break;
		}

		if( iCode >= 301 && iCode <= 303 )
		{
			sURLTmp = m_szLastURL;
			continue;
		}
		break;
	}while(true);

	return false;
}

bool CHttpDownloader::parseURL()
{
	if( m_sURL.empty() )
	{
		return false;
	}

	{
		string::size_type sLen = m_sURL.length();

		string::size_type st1 = m_sURL.find("//",0);
		if( st1 == string::npos )
		{
			return false;
		}

		string::size_type st2 = m_sURL.find("/",st1+2);

		if( st2 != string::npos )
		{
			if( m_sRelativePath.empty() )
			{
				m_sRelativePath = m_sURL.substr(st2,sLen - st2);
			}
		}		
	}
#ifdef _WINDOWS
	MBS2EncodeURL(m_sURL,m_sEncodeURL);
#else
	CCharsetConvertUtil::MBS2EncodeURL(m_sURL,m_sEncodeURL);
#endif//_WINDOWS
		string::size_type stURLLen = m_sEncodeURL.length();

	if( stURLLen <=7 )
	{
		return false;
	}

	{
		string sTmp = m_sEncodeURL.substr(0,7);

#ifdef _WINDOWS
		if( _strnicmp(sTmp.c_str(),"http://",7) != 0 )
		{
			return false;
		}
#else
		if( strncasecmp(sTmp.c_str(),"http://",7) != 0 )
		{
			return false;
		}
#endif//_WINDOWS
	}

	{
		string::size_type stTmp = m_sEncodeURL.find("/",7);

		if( stTmp == string::npos )
		{
			m_sGet = "/";
			m_sHost = m_sEncodeURL.substr(7,stURLLen-7);
		}
		else
		{
			m_sGet = m_sEncodeURL.substr(stTmp,stURLLen-stTmp);
			m_sHost = m_sEncodeURL.substr(7,stTmp-7);
		}
	}

	{
		string::size_type stHostLen = m_sHost.length();

		string::size_type st2 = m_sHost.find(":");

		string sHostname;
		if( st2!=string::npos )
		{
			m_sHostname = m_sHost.substr(0,st2);
			string sPort = m_sHost.substr(st2+1,stHostLen-st2-1);
			m_wPort = atoi(sPort.c_str());	
		}
		else
		{
			m_sHostname = m_sHost;
			m_wPort = 80;
		}
	}

	if( m_wPort && !m_sHostname.empty() )
	{
		return true;
	}

	return false;
}

#ifdef _WINDOWS
void CHttpDownloader::MBS2EncodeURL(const string& sURL,OUT string& sEncodeURL)
{
	sEncodeURL = sURL;

	{
		bool bNeedEncode = false;
		string::size_type nLen = sURL.length();
		string::const_iterator it = sURL.begin();
		for(; it != sURL.end(); ++it )
		{
			if( ((*it) & 0x80) || ((*it) == 0x20 ) )
			{
				bNeedEncode = true;
				break;
			}
		}

		if( bNeedEncode == false )
		{
			return;
		}
	} 	 

	{
		char szBuf[2048];
		memset(szBuf,0,2048);

		int nCurr = 0;
		string::const_iterator it = sURL.begin();
		for(; it != sURL.end();  )
		{
			char c = (*it);
			// 			if( c == '_' )
			// 			{
			// 				string sTmp(sURL.begin(),it);
			// 				szBuf[nCurr++] = c;	
			// 				++it;
			// 			}
			// 			else 
			if( c == 0x2B )//+
			{
				szBuf[nCurr++] = c;	
				++it;
			}
			else if( c == 0x26)//&
			{
				szBuf[nCurr++] = c;	
				++it;
			}
			else if( c == 0x20 )//空格
			{
				char szTmp[] = "%20";
				int nLenTmp = 3;
				memcpy(szBuf+nCurr,szTmp,6);
				nCurr+= 3;
				++it;
			}
			else if( c>0x20 && c< 0x7f )
			{
				szBuf[nCurr++] = c;
				++it;
			}
			else if( c < 0x20 && c > 0 )//不知怎么办
			{
				szBuf[nCurr++] = c;
				++it;
			}
			else
			{
				char szTmp[1024];
				memset(szTmp,0,1024);
				int iCurrTmp = 0;

				int nCntTmp = 0;
				do 
				{
					char cTmp = *it;
					if( nCntTmp % 2 == 0 )
					{
						if( cTmp < 0x7f && cTmp > 0)
						{
							break;
						}
					}

					szTmp[iCurrTmp++] = cTmp;

					++it;

					++nCntTmp;
				} while (it != sURL.end());

				szTmp[iCurrTmp] = 0;

				char utf8Buffer[4096];

				//将szTmp转换为UTF8
				wchar_t utf16Buffer[4096];
				int l_Count = 0;
				l_Count = ::MultiByteToWideChar( CP_ACP, 0,szTmp, -1, utf16Buffer, 4096 );
				utf16Buffer[l_Count] = 0;
				l_Count = ::WideCharToMultiByte( CP_UTF8, 0, utf16Buffer, -1, utf8Buffer, 4096, NULL, NULL );
				utf8Buffer[l_Count] = 0;

				boost::format fm("%%%2X");
				for( int i = 0; i< (l_Count-1); ++i )
				{
					fm.clear();					
					fm % ((int)(utf8Buffer[i]) & 0xff );
					memcpy(szBuf+nCurr,fm.str().c_str(),3);
					nCurr += 3;
				}			
			}			
		}

		szBuf [nCurr] = 0;

		sEncodeURL = szBuf;
	}
}
#endif//_WINDOWS

