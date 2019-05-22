
#ifndef __AY_EXCHANGEKEY_H__
#define __AY_EXCHANGEKEY_H__

#include "DHCryptLib.h"
#include "DiffieHellman.h"
#include "protocol_exchangekey.h"

class CAYExchangeKeyClient
{
public:
    CAYExchangeKeyClient();
    ~CAYExchangeKeyClient();
public:
    bool IsExchangeKey(){return m_isExchangeKey;}
    bool SetExchangedKey(uint8* key_data, int key_size);
    int EncryMsg(uint8* msg_buff, int msg_len);
    int DecryMsg(uint8* msg_buff, int msg_len);
public:
    int BuildExchangeKeyRequest(protocol::ExchangeKeyRequest& msg_req);
    int OnExchangeKeyResponse(const protocol::ExchangeKeyResponse& msg_resp);
private:
    int GetExchangeKey(uint8* key_buff, int key_size);
private:
    bool m_isExchangeKey;
    int m_nKeyPos;
private:
    DiffieHellman* dhe_;
    uint8 m_szKeyBuff[64];
    int m_nKeySize;
};

class CAYExchangeKeyServer
{
public:
    CAYExchangeKeyServer();
    ~CAYExchangeKeyServer();
public:
    bool IsExchangeKey(){return m_isExchangeKey;}
    bool SetExchangedKey(uint8* key_data, int key_size);
    int EncryMsg(uint8* msg_buff, int msg_len);
    int DecryMsg(uint8* msg_buff, int msg_len);
public:
    int OnExchangeKeyRequest(const protocol::ExchangeKeyRequest& msg_req);
    int BuildExchangeKeyResponse(protocol::ExchangeKeyResponse& msg_resp);
private:
    virtual int GetExchangeKey(uint8* key_buff, int key_size);
private:
    bool m_isExchangeKey;
    int m_nKeyPos;
private:
    DiffieHellman* dhe_;
    uint8 m_szKeyBuff[64];
    int m_nKeySize;
};

class CAYExchangeKeyTest
{
public:
    CAYExchangeKeyTest(){}
    ~CAYExchangeKeyTest(){}
    int Test();
private:
    CAYExchangeKeyClient client;
    CAYExchangeKeyServer server;
};

#endif /* __AY_EXCHANGEKEY_H__ */
