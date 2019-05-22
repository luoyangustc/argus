#include "RedisAccess.h"

#include "base/include/GetTickCount.h"
#include "base/include/logging_posix.h"
#include "base/include/ConfigHelper.h"

namespace
{
    // Hash Function
    static unsigned int HashCallback(const char *str) 
    {
        unsigned int hash = 0;
        for (int iter = 0; *str; ++iter)
        {
            if ((iter & 1) == 0)
            {
                hash ^= ((hash << 7) ^ (*str++) ^ (hash >> 3));
            } 
            else
            {
                hash ^= (~((hash << 11) ^ (*str++) ^ (hash >> 5)));
            }
        }
        return hash & 0x7FFFFFFF;
    }

}  // namespace

RedisAccess:: RedisAccess()
    : xredis_client_()
    , redis_dbindex_(&xredis_client_)
    , redis_cache_type_(kRedisCacheTypeMax)
    , last_heartbeat_tick_(0)
{
}

RedisAccess::~RedisAccess()
{
}

bool RedisAccess::Initialize(unsigned int cache_type)
{
    redis_cache_type_ = cache_type;

    RedisNode redis_node;
    ReadRedisConfig(&redis_node);  

    bool bRet = false;
    do
    {
        if (! xredis_client_.Init(kRedisCacheTypeMax))
        {
            Fatal("xredis_client_ Init error\n");    
            break;
        }

        if (! xredis_client_.ConnectRedisCache(&redis_node,1,redis_cache_type_))
        {
            Error("Redis client connect to server failed \n");    
            break;
        }
        Debug("connect to redis server ok\n");

        // redis client connect to server successfully
        bRet = true;
    }while(false);

    // please don't forget to delete [host/passwd] heap memory !
    delete[]  redis_node.host;
    delete[]  redis_node.passwd;

    return bRet;
}

bool RedisAccess::set(const std::string& key,const std::string& value)
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    //this is thread-safe !!!
    if (xredis_client_.set(redis_dbindex_, key.c_str(), value.c_str()))
    {
        return true;
    }
    else
    {
        Error("set error(%s) for key(%s),value(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str(),value.c_str());
        return false;
    }
}

bool RedisAccess::get(const std::string& key,std::string& value)
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.get(redis_dbindex_, key, value))
    {
        return true;
    }
    else
    {
        Error("get error(%s) for key(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str());
        return false;
    }
}

bool RedisAccess::hmset(const std::string& key, const VDATA& vData)
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.hmset(redis_dbindex_, key, vData))
    {
        return true;
    }
    else
    {
        Error("hmset error(%s) for key(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str());
        return false;
    }
}

bool RedisAccess::hset(const std::string& key,
    const std::string& field,
    const std::string& value,
    int64_t& retval)
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.hset(redis_dbindex_, key, field, value, retval))
    {
        return true;
    }
    else
    {
        Error("hset error(%s) for key(%s),field(%s),value(%s)\n",
            redis_dbindex_.GetErrInfo(),key.c_str(),field.c_str(),value.c_str());
        return false;
    }
}

bool RedisAccess::hmget( const std::string& key, const KEYS& field, ArrayReply& array )
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.hmget(redis_dbindex_, key, field, array))
    {
        return true;
    }
    else
    {
        Error("hmget error(%s) for key(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str());
        return false;
    }
}

bool RedisAccess::hget( const std::string& key, const std::string& field, std::string& value )
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.hget(redis_dbindex_, key, field, value))
    {
        return true;
    }
    else
    {
        Error("hget error(%s) for key(%s),field(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str(),field.c_str());
        return false;
    }
}

bool RedisAccess::hgetall( const std::string& key, ArrayReply& array )
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.hgetall(redis_dbindex_, key, array))
    {
        return true;
    }
    else
    {
        Error("hgetall error(%s) for key(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str());    
        return false;
    }
}

bool RedisAccess::rpop( const std::string& key, std::string& value )
{   
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.rpop(redis_dbindex_, key, value))
    {
        return true;
    }
    else
    {
        Error("rpop error(%s) for key(%s),value(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str(),value.c_str());  
        return false;
    }
}

bool RedisAccess::lpop( const std::string& key, std::string& value )
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);
    if (xredis_client_.lpop(redis_dbindex_, key, value))
    {
        return true;
    }
    else
    {
        Error("lpop error(%s) for key(%s),value(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str(),value.c_str());  
        return false;
    }
}

bool RedisAccess::rpush( const std::string& key, const VALUES& vValue, int64_t& length )
{    
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.rpush(redis_dbindex_, key, vValue, length))
    {
        return true;
    }
    else
    {
        Error("rpush error(%s) for key(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str());
        return false;
    }
}

bool RedisAccess::rpushx( const std::string& key, const std::string& value, int64_t& length )
{    
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.rpushx(redis_dbindex_, key, value, length))
    {
        return true;
    }
    else
    {
        Error("rpushx error(%s) for key(%s),value(%s)\n",redis_dbindex_.GetErrInfo(),key.c_str(),value.c_str());  
        return false;
    }
}

bool RedisAccess::lrange( const std::string& key, int64_t start, int64_t end, ArrayReply& array )
{    
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.lrange(redis_dbindex_, key, start, end, array))
    {
        return true;
    }
    else
    {
        Error("lrange error(%s) for key(%s),start(%d),end(%d)\n",redis_dbindex_.GetErrInfo(),key.c_str(),start,end);  
        return false;
    }
}

bool RedisAccess::llen(const std::string& key,int64_t& count)
{
    redis_dbindex_.CreateDBIndex(key.c_str(), HashCallback, redis_cache_type_);

    if (xredis_client_.llen(redis_dbindex_, key, count))
    {
        return true;
    }
    else
    {
        Error("llen error(%s) for key(%s),count(%d)\n",redis_dbindex_.GetErrInfo(),key.c_str(),count);  
        return false;
    }
}

bool RedisAccess::sort(
        ArrayReply& array, 
        const std::string& key, 
        const char* by,
        LIMIT *limit, 
        bool alpha, 
        const FILEDS* get, 
        const SORTODER order, 
        const char* destination )
{
    return false;
}

void RedisAccess::Keepalive(int heartbeat_cycle)
{  
    const int time_interval = GetTickCount() - last_heartbeat_tick_;
    if (time_interval > heartbeat_cycle)
    {
        last_heartbeat_tick_ = GetTickCount();
        Info("Heartbeat between Redis client and Server at %d \n", last_heartbeat_tick_);  
        xredis_client_.Keepalive();    
    }
}

void RedisAccess::ReadRedisConfig(RedisNode * pRedisNode)
{
    assert(pRedisNode != NULL);

    // FIXME : Note that the std::string format in the configuration file !
    const std::string config_file = CConfigHelper::get_default_config_filename();
    pRedisNode->dbindex = GetPrivateProfileInt("redis", 
        "index",
        0,
        config_file.c_str());

    char* pHost = new char[256];  //FIXME:256 bytes may be enough
    GetPrivateProfileString("redis",
        "host",
        "127.0.0.1",
        pHost,
        256/*256 bytes*/, 
        config_file.c_str());
    pRedisNode->host = pHost;

    pRedisNode->port = GetPrivateProfileInt("redis",
        "port",
        6379,
        config_file.c_str());

    char* pPassword = new char[256];  //FIXME:256 bytes may be enough
    GetPrivateProfileString("redis",
        "password",
        "123456",
        pPassword,
        256/*256 bytes*/, 
        config_file.c_str());
    pRedisNode->passwd = pPassword;

    pRedisNode->poolsize = GetPrivateProfileInt("redis",
        "poolsize",
        8,
        config_file.c_str());

    pRedisNode->timeout = GetPrivateProfileInt("redis",
        "timeout",
        5,
        config_file.c_str());

    pRedisNode->role = GetPrivateProfileInt("redis",
        "role",
        0,
        config_file.c_str());
}