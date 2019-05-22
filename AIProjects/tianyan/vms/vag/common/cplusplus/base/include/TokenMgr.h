#ifndef __TOKEN_MGR_H__
#define __TOKEN_MGR_H__

#include <map>
#include <boost/thread.hpp>
#include <boost/bimap/bimap.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/shared_ptr.hpp>

#include "typedefine.h"
#include "public_key.h"
#include "private_key.h"
#include "HostInfo.h"
#include "tick.h"
#include "DeviceID.h"

using namespace std;

struct token_info
{
	token_t token;
	boost::shared_ptr<c3_crypt_key::STokenSource> p_token_source;
	mutable bool valid;
	tick_t create_tick;
	token_info()
	{
		valid = true;
		create_tick = get_current_tick();
	}
	void disable() const
	{
		valid = false;
	}

	bool is_valid() const
	{
		return valid;
	}
	const token_info& operator=(const token_info& right)
	{
		token = right.token;
		p_token_source = right.p_token_source;
		valid = right.valid;
		return *this;
	}
	bool operator<(const token_info& right)const
	{
		return token < right.token;
	}
	bool operator==(const token_info& right)const
	{
		return token == right.token;
	}
	bool operator!=(const token_info& right)const
	{
		return token != right.token;
	}

	bool operator<=(const token_info& right)const
	{
		return token <= right.token;
	}

	bool operator>(const token_info& right)const
	{
		return token > right.token;	
	}

	bool operator>=(const token_info& right)const
	{
		return token >= right.token;
	}
};

typedef boost::bimaps::bimap<
	token_info,
	boost::bimaps::multiset_of< uint32 >
>TokenMap;

//typedef TokenMap::value_type Token_Info;

class CTokenMgr
{
public:
	CTokenMgr(void);
	~CTokenMgr(void);

    void Update();

    bool SetKey(const string& access_key, const string& secret_key);
    bool TokenGen(IN const string& access_key, IN const string& plain_text, OUT string& token);
    bool TokenVerify(IN const string& token, IN const string& token_type, OUT string& meta_data);
    
    //会话服务token
    bool SessionDeviceToken_Auth( IN const string& token, IN const string& device_id );
    bool SessionUserToken_Auth( IN const string& token, IN const string& user_id );
    bool SessionDeviceToken_Gen( IN const string& device_id, IN uint32 duration, IN const string& access_key, OUT string& token );
    bool SessionUserToken_Gen( IN const string& user_id, IN uint32 duration, IN const string& access_key, OUT string& token );

    //流服务token
    bool StreamDeviceToken_Auth( IN const string& token, IN const string& device_id, IN int channel_id );
    bool StreamUserToken_Auth( IN const string& token, IN const string& user_id, IN const string& device_id, IN int channel_id );
    bool StreamDeviceToken_Gen( IN const string& device_id, IN int channel_id, IN uint32 duration, IN const string& access_key, OUT string& token );
    bool StreamUserToken_Gen( IN const string& user_id, IN const string& device_id, IN int channel_id, IN uint32 duration, IN const string& access_key, OUT string& token );
public:
    bool Test();
private:
	boost::recursive_mutex lock_;
    map<string, string> key_tables_;
	TokenMap tokens_;
};

typedef boost::shared_ptr<CTokenMgr> CTokenMgr_ptr;

#endif //__TOKEN_MGR_H__

