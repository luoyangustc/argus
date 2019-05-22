#ifndef __TYPEDEFINE_H__
#define __TYPEDEFINE_H__

#ifdef _WINDOWS
//#define uint64 unsigned  __int64
typedef unsigned  __int64 uint64;
#else
#define uint64 unsigned long long
#endif

#ifdef _WINDOWS
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
#else
#define uint32 unsigned int
#define uint16 unsigned short
#define uint8 unsigned char
#endif

#ifdef _WINDOWS
//#define int64 __int64
typedef __int64 int64;
#else
#define int64 long long
#endif


#ifdef _WINDOWS
typedef int int32; 
typedef  short int16;
typedef  signed char int8; 
#else
#define int32 int
#define int16 short
#define int8 signed char
#endif

#include <string.h>

#define nil	0

typedef uint16 version_t[4];

enum EAYDevice_Type
{
	ay_device_unknown  = 0,
	ay_device_dvr	   = 1,
	ay_device_nvr      = 2,
	ay_device_ipcam    = 3,
};

typedef struct __device_id_t
{
	__device_id_t()
	{
		device_id_length = 0;
		memset(device_id,0,sizeof(device_id));
	}
	const __device_id_t& operator=(const __device_id_t& right)
	{
		memcpy(this,&right,sizeof(__device_id_t));
		return *this;
	}
	bool operator<(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))<0?true:false;	
	}
	bool operator==(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))==0?true:false;	
	}
	bool operator!=(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))!=0?true:false;	
	}

	bool operator<=(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))<=0?true:false;	
	}

	bool operator>(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))>0?true:false;	
	}

	bool operator>=(const __device_id_t& right)const
	{
		return memcmp(this,&right,sizeof(__device_id_t))>=0?true:false;	
	}
	uint8 device_id_length;
	uint8 device_id[21];
}device_id_t;


typedef struct __token_t
{
	__token_t()
	{
		token_bin_length = 0;
		memset(token_bin,0,sizeof(token_bin));
	}

	const __token_t& operator=(const __token_t& right)
	{
		memcpy(this,&right,sizeof(__token_t));
		return *this;
	}
	bool operator<(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))<0?true:false;	
	}
	bool operator==(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))==0?true:false;	
	}
	bool operator!=(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))!=0?true:false;	
	}

	bool operator<=(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))<=0?true:false;	
	}

	bool operator>(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))>0?true:false;	
	}

	bool operator>=(const __token_t& right)const
	{
		return memcmp(this,&right,sizeof(__token_t))>=0?true:false;	
	}
	uint16 token_bin_length;
	uint8 token_bin[256];
}token_t;

typedef struct __c3_error_t{
	__c3_error_t()
	{
		error_code = 0;
		memset(error_description,0,sizeof(error_description));
	}
	int32	error_code;
	char	error_description[256];
}c3_error_t;

#include <string>
#include <map>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
using namespace std;

struct SHttpUrl
{
    string page_;
    map<string,string> params_;
};

class CHttpHeader
{
public:
    CHttpHeader(void){
        method_ = "GET";
        code_ = 200;
        url_ = "/";
        is_request = true;
        url_detail_.page_ = "";
        url_detail_.params_.clear();
    }
    ~CHttpHeader(void){}

    void clear()
    {
        method_ = "GET";
        code_ = 200;
        url_ = "/";
        is_request = true;
        map_key_values_.clear();
        url_detail_.page_ = "";
        url_detail_.params_.clear();
    }
public:
    std::string method_;
    bool is_request;
    int code_;
    std::string url_;    
    SHttpUrl url_detail_;
    std::map<string,string> map_key_values_;
};
typedef boost::shared_ptr<CHttpHeader> CHttpHeader_ptr;

struct SHttpRequestPara{
    CHttpHeader_ptr header_detail;
    boost::shared_array<uint8> header_data;
    uint32 header_data_len;
    boost::shared_array<uint8> content_data;
    uint32 content_data_len;
};


/*
struct SHttpRequestPara{
	char* szHeader;
	int header_len;
	unsigned char* szBody;
	int body_len;
};
*/

typedef boost::shared_ptr<SHttpRequestPara> SHttpRequestPara_ptr;

struct SHttpResponsePara{
	boost::shared_array<uint8> pContent;
	uint32 content_len;
	std::string ret_code;
	std::string content_type;
	std::string location;
    std::string user_config;
	bool keep_alive;
	SHttpResponsePara()
	{
		content_len = 0;
		keep_alive = false;
	}
};

typedef boost::shared_ptr<SHttpResponsePara> SHttpResponsePara_ptr;

#define LADJUST_RATE(rate) ((rate<=384)?384:((rate<=500)?500:((rate<=700)?700:((rate<=1000)?1000:(1000)))))

#endif //__TYPEDEFINE_H__

