
#include <stdio.h>
#include "http_header_util.h"
//#include "HttpParserUtil.h"

bool CHttpHeaderUtil::ParseHttpURL(const char* szURL, string & strObject, string& strHost, uint16& usPort)
{
    return false;
}

bool CHttpHeaderUtil::MakeHttpHeaderRequest( IN OUT char szBuff[],IN int buff_len ,const char* szMethod, const char* szURL, const char* szContentType, int content_length, bool is_close)
{
    int total_len = 0;
    int temp_len = 0;
    char temp_buff[1024];
    temp_len = sprintf(temp_buff, "%s %s HTTP/1.1\r\n", szMethod, szURL);
    if( temp_len >= (buff_len-1) )
    {
        return false;
    }
    memcpy(szBuff, temp_buff, temp_len);
    szBuff[temp_len] = '\0';

    if( (int)(strlen(szBuff)+strlen("Accept: text/html\r\n") ) >= (buff_len-1) )
    {
        return false;
    }
    strcat(szBuff, "Accept: text/html\r\n");

    if( (int)(strlen(szBuff)+strlen("Accept-Encoding: gzip, deflate\r\n") ) >= (buff_len-1) )
    {
        return false;
    }
    strcat(szBuff, "Accept-Encoding: gzip, deflate\r\n");

    if(is_close)
    {
        if( (int)(strlen(szBuff)+strlen("Connection: close\r\n") ) >= (buff_len-1) )
        {
            return false;
        }
        strcat(temp_buff, "Connection: close\r\n");
    }
    else
    {
        if( (int)(strlen(szBuff)+strlen("Connection: keep-alive\r\n") ) >= (buff_len-1) )
        {
            return false;
        }
        strcat(temp_buff, "Connection: keep-alive\r\n");
    }

    if( szContentType )
    {
        temp_len = sprintf(temp_buff, "Content-Type: %s\r\n", szContentType);
        if( (int)(strlen(szBuff)+temp_len) >= (buff_len-1) )
        {
            return false;
        }
        temp_buff[temp_len] = '\0';
        strcat(szBuff, temp_buff);
    }

    if( content_length > 0 )
    {
        temp_len = sprintf(temp_buff, "Content-Length: %d\r\n", content_length);
        if( (int)(strlen(szBuff)+temp_len) >= (buff_len-1) )
        {
            return false;
        }
        szBuff[temp_len] = '\0';
        strcat(szBuff, temp_buff);
    }

    if( (int)(strlen(szBuff)+2) > (buff_len-1) )
    {
        return false;
    }

    strcat(szBuff, "\r\n");

    return true;
}

bool MakeHttpHeaderResponse( 
		OUT char szBuff[] ,IN int buff_len,
		const char * stauts_desc,int content_length,
		bool is_close, const char* content_type,const char* location,const char* user_config)
{
    int total_len = 0;
    int temp_len = 0;
    char temp_buff[1024];
    temp_len = sprintf(temp_buff, "HTTP/1.1 %s\r\n", stauts_desc);
    if( temp_len >= (buff_len-1) )
    {
        return false;
    }
    memcpy(szBuff, temp_buff, temp_len);
    szBuff[temp_len] = '\0';

    if( is_close )
    {
        if( (int)(strlen(szBuff)+strlen("Connection: close\r\n") ) >= (buff_len-1) )
        {
            return false;
        }
        strcat(temp_buff, "Connection: close\r\n");
    }
    else
    {
        if( (int)(strlen(szBuff)+strlen("Connection: keep-alive\r\n") ) >= (buff_len-1) )
        {
            return false;
        }
        strcat(temp_buff, "Connection: keep-alive\r\n");
    }

    if( content_type )
    {
        temp_len = sprintf(temp_buff, "Content-Type: %s\r\n", content_type);
        if( (int)(strlen(szBuff)+temp_len) >= (buff_len-1) )
        {
            return false;
        }
        temp_buff[temp_len] = '\0';
        strcat(szBuff, temp_buff);
    }

    if( content_length>0 )
    {
        temp_len = sprintf(temp_buff, "Content-Length: %d\r\n", content_length);
        if( (int)(strlen(szBuff)+temp_len) >= (buff_len-1) )
        {
            return false;
        }
        szBuff[temp_len] = '\0';
        strcat(szBuff, temp_buff);
    }

    if( location )
    {
        temp_len = sprintf(temp_buff, "Location: %s\r\n", location);
        if( (int)(strlen(szBuff)+temp_len) >= (buff_len-1) )
        {
            return false;
        }
        szBuff[temp_len] = '\0';
        strcat(szBuff, temp_buff);
    }

    if( (int)(strlen(szBuff)+2) > (buff_len-1) )
    {
        return false;
    }

    strcat(szBuff, "\r\n");

    return true;
}

bool CHttpHeaderUtil::ParseHttpHeader(char * szHeader,int header_len,OUT CHttpHeader_ptr& pHeader )
{
	#if 0
    SHttpMessage msg;
    if( !CHttpParserUtil::parser_execute(szHeader, header_len, msg) )
    {
        return false;
    }
    
    *pHeader = msg.header_;
	#endif
    return true;
}

bool CHttpHeaderUtil::MakeURL( const char * szKey,OUT char szBuff[] ,IN int buff_len)
{
    return false;
}

bool CHttpHeaderUtil::ParseURLParams(const char* url, string& page, map<string,string>& params)
{
    return false;
}


