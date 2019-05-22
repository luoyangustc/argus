#ifndef STATUS_SERVER_REDISACCESS_H
#define STATUS_SERVER_REDISACCESS_H

#include <boost/noncopyable.hpp>
#include <boost/thread/detail/singleton.hpp>
#include <boost/thread/thread.hpp>

#include "third_party/redis/include/xRedisClient.h"

class RedisAccess : boost::noncopyable
{
public:
    enum RedisCacheType { kRedisCacheTypeClient = 0, kRedisCacheTypeMax};

    RedisAccess();
    ~RedisAccess(); //The class will not be inherited

    bool Initialize(unsigned int cache_type = kRedisCacheTypeClient);

    bool set(const std::string& key,const std::string& value);
    bool get(const std::string& key,std::string& value);
    bool hmset(const std::string& key, const VDATA& vData);	
    bool hset(const std::string& key,
        const std::string& field,
        const std::string& value, 
        int64_t& retval);	
    bool hmget(const std::string& key,
        const KEYS& field,
        ArrayReply& array);	
    bool hget(const std::string& key,
        const std::string& field, 
        std::string& value);
    bool hgetall(const std::string& key, ArrayReply& array);
    bool rpop(const std::string& key, std::string& value);
    bool lpop(const std::string& key, std::string& value);    
    bool rpush(const std::string& key,
        const VALUES& vValue,
        int64_t& length);	
    bool rpushx(const std::string& key,
        const std::string& value,
        int64_t& length);
    bool lrange(const std::string& key,
        int64_t start,
        int64_t end,
        ArrayReply& array);
    bool llen(const std::string& key,int64_t& count);    
    bool sort(ArrayReply& array, 
        const std::string& key, 
        const char* by = NULL,
        LIMIT *limit = NULL, 
        bool alpha = false, 
        const FILEDS* get = NULL, 
        const SORTODER order = ASC, 
        const char* destination = NULL);

    void Keepalive(int heartbeat_cycle);

private:
    void ReadRedisConfig(RedisNode* pRedisNode);

    xRedisClient xredis_client_;
    RedisDBIdx redis_dbindex_;
    unsigned int redis_cache_type_;
    // The heartbeat timestamp between redis client and server
    int last_heartbeat_tick_;    
};

typedef boost::detail::thread::singleton<RedisAccess> RedisAccessSingleton;

#endif  // STATUS_SERVER_REDISACCESS_H
