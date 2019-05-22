#ifndef __HTTP_HEADER_UTIL_H__
#define __HTTP_HEADER_UTIL_H__

#pragma once

#include <string>
#include <map>
using namespace std;

#include "typedef_win.h"
#include "typedefine.h"
#include <map>
#include <boost/shared_ptr.hpp>
#if 0
class CHttpHeader
{
public:
    CHttpHeader(void)
    {
        method_ = "GET";
        code_ = 200;
        url_ = "/";
        is_request = true;
    }

    ~CHttpHeader(void){}
    
    void clear()
    {
        method_ = "GET";
        code_ = 200;
        url_ = "/";
        is_request = true;
        map_key_values_.clear();
    }
public:
    string method_;
    int code_;
    string url_;
    bool is_request;
    map<string,string> map_key_values_;
};

typedef boost::shared_ptr<CHttpHeader> CHttpHeader_ptr;
#endif

class CHttpHeaderUtil
{
public:
	CHttpHeaderUtil(void){}
	~CHttpHeaderUtil(void){}

	static bool ParseHttpURL(const char* szURL, string & strObject, string& strHost, uint16& usPort);
	static bool MakeHttpHeaderRequest( IN OUT char szBuff[],IN int buff_len ,const char* szMethod, const char* szURL, const char* szContentType, int content_length, bool is_close);
	static bool MakeHttpHeaderResponse( 
		OUT char szBuff[] ,IN int buff_len,
		const char * stauts_desc,int content_length,
		bool is_close,/*const char * cookie*/const char* content_type,const char* location,const char* user_config);

	static bool ParseHttpHeader(char * szHeader,int header_len,OUT CHttpHeader_ptr& pHeader );
	static bool MakeURL( const char * szKey,OUT char szBuff[] ,IN int buff_len);

	static bool ParseURLParams(const char*url,string& page,map<string,string>& params);
};

#endif //__HTTP_HEADER_UTIL_H__

