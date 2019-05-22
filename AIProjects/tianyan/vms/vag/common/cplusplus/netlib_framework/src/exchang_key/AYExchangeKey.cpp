#include <stdlib.h>
#include <assert.h>
#include "AYCrypt.h"
#include "AYExchangeKey.h"

//**************************/
//** CAYExchangeKeyClient
//**************************/
CAYExchangeKeyClient::CAYExchangeKeyClient()
{
    dhe_ = NULL;

    m_nKeySize = 0;
    m_isExchangeKey = false;
    m_nKeyPos = 0;
}

CAYExchangeKeyClient::~CAYExchangeKeyClient()
{
    if (dhe_)
    {
        delete dhe_;
        dhe_ = NULL;
    }
}

bool CAYExchangeKeyClient::SetExchangedKey(uint8* key_data, int key_size)
{
    if( !key_data || (key_size>sizeof(m_szKeyBuff)) )
    {
        return false;
    }

    memcpy(m_szKeyBuff, key_data, key_size);
    m_nKeySize = key_size;

    m_isExchangeKey = true;

    return true;
}

int CAYExchangeKeyClient::EncryMsg(uint8* msg_buff, int msg_len)
{
    if( !m_isExchangeKey )
    {
        return 0;
    }
    return CAYCrypt::EncryTcpMsg(msg_buff, msg_len, m_nKeyPos, m_szKeyBuff, m_nKeySize );
}

int CAYExchangeKeyClient::DecryMsg(uint8* msg_buff, int msg_len)
{
    if( !m_isExchangeKey )
    {
        return 0;
    }

    return CAYCrypt::DecryTcpMsg(msg_buff, msg_len, m_nKeyPos, m_szKeyBuff, m_nKeySize );
}

int CAYExchangeKeyClient::BuildExchangeKeyRequest(protocol::ExchangeKeyRequest& msg_req)
{
    int ret = 0;
    do 
    {
        if( dhe_ )
        {
            delete dhe_;
        }
        dhe_ = new DiffieHellman(8);
        if( !dhe_ )
        {
            ret = -1;
            break;
        }

        if( dhe_->MakePrime() < 0 )
        {
            ret = -2;
            break;
        }

        if( dhe_->ComputesA() < 0 )
        {
            ret = -3;
            break;
        }

        //mask 0x01
        msg_req.mask = 0x01;
        //dhe_->Get_A_P();
        UINT key_A_length = sizeof(msg_req.key_A);
        UINT key_P_length = sizeof(msg_req.key_P);
        if( dhe_->Get_A_P(msg_req.key_A, &key_A_length, msg_req.key_P, &key_P_length) < 0 )
        {
            ret = -4;
            break;
        }
        msg_req.key_A_length = key_A_length;
        msg_req.key_P_length = key_P_length;

        //mask 0x02
        msg_req.mask |= 0x02;
#if NO_EXCHANGE_DEBUG
        msg_req.except_algorithm = protocol::EExchangekey_Nothing;
        msg_req.algorithm_param = 0;
#else
        msg_req.except_algorithm = protocol::EExchangekey_Device_1;
        msg_req.algorithm_param = DEVICE_1_KEY_POS;
#endif

    } while (0);
    
    return ret;
}

int CAYExchangeKeyClient::OnExchangeKeyResponse(const protocol::ExchangeKeyResponse& msg_resp)
{
    int ret = -1;
    do 
    {
        if(!dhe_)
        {
            ret = -1;
            break;
        }

        if (msg_resp.resp_code != 0 )
        {
            ret = -2;
            break;
        }

        if(msg_resp.encry_algorithm ==  protocol::EExchangekey_Nothing)
        {
            ret = 0;
            break;
        }
        else if(msg_resp.encry_algorithm ==  protocol::EExchangekey_Device_1)
        {
            if (msg_resp.key_B_length > MAX_KEY_LENGTH/8)
            {
                ret = -3;
                break;
            }

            if ( dhe_->Set_B(msg_resp.key_B, msg_resp.key_B_length) < 0 )
            {
                ret = -4;
                break;
            }

            if( dhe_->ComputesS1() < 0 )
            {
                ret = -5;
                break;
            }

            uint8 szkey[64];
            uint16 key_size = msg_resp.key_size;
            if( dhe_->Get_S1(szkey, key_size) < 0 )
            {
                ret = -6;
                break;
            }

            (void)SetExchangedKey(szkey, key_size);
            m_nKeyPos = (int32)msg_resp.algorithm_param;

            ret = 0;
            break;
        }
        else // not support now!
        {
            ret = -7;
            break;
        }

    } while (0);
    
    return ret;
}

int CAYExchangeKeyClient::GetExchangeKey(uint8* key_buff, int key_size)
{
    return dhe_->Get_S1(key_buff, key_size);
}

//**************************/
//** CAYExchangeKeyServer
//**************************/
CAYExchangeKeyServer::CAYExchangeKeyServer()
{
    dhe_ = NULL;
    m_nKeySize = 0;
    m_isExchangeKey = false;
    m_nKeyPos = 0;
}

CAYExchangeKeyServer::~CAYExchangeKeyServer()
{
    if( dhe_ )
    {
        delete dhe_;
        dhe_ = NULL;
    }
}

bool CAYExchangeKeyServer::SetExchangedKey(uint8* key_data, int key_size)
{
    if( !key_data || (key_size>sizeof(m_szKeyBuff)) )
    {
        return false;
    }

    memcpy(m_szKeyBuff, key_data, key_size);
    m_nKeySize = key_size;

    m_isExchangeKey = true;

    return true;
}

int CAYExchangeKeyServer::EncryMsg(uint8* msg_buff, int msg_len)
{
    if( !m_isExchangeKey )
    {
        return 0;
    }
    return CAYCrypt::EncryTcpMsg(msg_buff, msg_len, m_nKeyPos, m_szKeyBuff, m_nKeySize );
}

int CAYExchangeKeyServer::DecryMsg(uint8* msg_buff, int msg_len)
{
    if( !m_isExchangeKey )
    {
        return 0;
    }

    return CAYCrypt::DecryTcpMsg(msg_buff, msg_len, m_nKeyPos, m_szKeyBuff, m_nKeySize );
}

int CAYExchangeKeyServer::BuildExchangeKeyResponse(protocol::ExchangeKeyResponse& msg_resp)
{
    int ret = 0;
    do 
    {
        if( !m_isExchangeKey )
        {
            msg_resp.mask = 0x01;
            msg_resp.resp_code = protocol::EN_SUCCESS;
            msg_resp.mask |= 0x02;
            msg_resp.encry_algorithm = protocol::EExchangekey_Nothing;
        }
        else
        {
            if( !dhe_ )
            {
                ret = -1;
                break;
            }

            msg_resp.mask = 0x01;
            msg_resp.resp_code = protocol::EN_SUCCESS;

            UINT key_B_length = sizeof(msg_resp.key_B);
            if( dhe_->Get_B(msg_resp.key_B, &key_B_length) < 0 )
            {
                ret = -2;
                break;
            }
            msg_resp.key_B_length = key_B_length;
            msg_resp.key_size = dhe_->GetKeySize();

            msg_resp.mask |= 0x02;
            msg_resp.encry_algorithm = protocol::EExchangekey_Device_1;
            if(m_nKeyPos==0)
            {
                msg_resp.algorithm_param = DEVICE_1_KEY_POS;
                m_nKeyPos = DEVICE_1_KEY_POS;
            }

            ret = 0;
            break;
        }
    } while (0);
    
    if(ret < 0)
    {
        msg_resp.mask = 0x02;
        msg_resp.resp_code = -1;
        //msg_resp.encry_algorithm = protocol::EExchangekey_Nothing;
        if(dhe_)
        {
            delete dhe_;
            dhe_ = NULL;
        }
    }

    return ret;
}

int CAYExchangeKeyServer::OnExchangeKeyRequest(const protocol::ExchangeKeyRequest& msg_req)
{
    int ret = -1;
    do
    {
        if( dhe_ )
        {
            delete dhe_;
            dhe_ = NULL;
        }

        if( !(msg_req.mask&0x01)
         || !(msg_req.mask&0x02) )
        {
            ret = -2;
            break;
        }

        if(msg_req.except_algorithm ==  protocol::EExchangekey_Nothing)
        {
            ret = 0;
            break;
        }
        else if(msg_req.except_algorithm ==  protocol::EExchangekey_Device_1)
        {
            dhe_ = new DiffieHellman(8);
            if( !dhe_ )
            {
                ret = -3;
                break;
            }
            /*
            if( dhe_->MakePrime() < 0 )
            {
                ret = -4;
                break;
            }

            if( dhe_->ComputesA() < 0 )
            {
                ret = -5;
                break;
            }*/

            if( dhe_->Set_A_P(msg_req.key_A, msg_req.key_A_length, msg_req.key_P, msg_req.key_P_length) < 0 )
            {
                ret = -6;
                break;
            }
        
            if( dhe_->ComputesB() < 0 ) 
            {
                ret = -7;
                break;
            }
            if( dhe_->ComputesS2() < 0 ) 
            {
                ret = -8;
                break;
            }

            uint8 szkey[64];
            int32 key_size = dhe_->GetKeySize();
            if( dhe_->Get_S2(szkey, key_size) < 0 )
            {
                ret = -5;
                break;
            }

            (void)SetExchangedKey(szkey, key_size);

            m_nKeyPos = (int32)msg_req.algorithm_param;

            ret = 0;
            break;
        }
        else //not support now!
        {
            ret = -9;
            break;
        }

    } while (0);

    //if failed, release dhe_
    if (ret < 0)
    {
        if(dhe_)
        {
            delete dhe_;
            dhe_ = NULL;
        }
    }

    return ret;
}

int CAYExchangeKeyServer::GetExchangeKey(uint8* key_buff, int key_size)
{
    return dhe_->Get_S2(key_buff, key_size);
}


int CAYExchangeKeyTest::Test()
{
    char szKeyReq[512];
    char szKeyResp[512];

    //1. client send exchange key request
    CDataStream req_ds(szKeyReq, sizeof(szKeyReq));
    int ret = 0;
    {
        protocol::ExchangeKeyRequest msg_key_req;
        ret = client.BuildExchangeKeyRequest(msg_key_req);
        assert(ret>=0);

        protocol::MsgHeader msg_head;
        msg_head.msg_id = protocol::MSG_ID_EXCHANGE_KEY;
        msg_head.msg_type = protocol::MSG_TYPE_REQ;
        msg_head.msg_seq = 1;

        req_ds << msg_head;
        req_ds << msg_key_req;
        *(uint16*)req_ds.getbuffer() = req_ds.size();
    }

    //2. server recveive exchange key request
    {
        CDataStream recvds_req((char*)req_ds.getbuffer(), req_ds.size());
        protocol::MsgHeader msg_head;
        protocol::ExchangeKeyRequest msg_key_req;
        recvds_req>>msg_head;
        recvds_req>>msg_key_req;
        ret = server.OnExchangeKeyRequest(msg_key_req);
        assert(ret>=0);
    }
    
    //3. server send exchange key response
    CDataStream resp_ds(szKeyResp, sizeof(szKeyResp));
    {
        protocol::ExchangeKeyResponse msg_key_resp;
        ret = server.BuildExchangeKeyResponse(msg_key_resp);
        assert(ret>=0);

        protocol::MsgHeader msg_head;
        msg_head.msg_id = protocol::MSG_ID_EXCHANGE_KEY;
        msg_head.msg_type = protocol::MSG_TYPE_RESP;
        msg_head.msg_seq = 1;

        resp_ds << msg_head;
        resp_ds << msg_key_resp;
        *(uint16*)resp_ds.getbuffer() = resp_ds.size();
    }
    
    //4. client receive exchange key response
    {
        CDataStream recvds_resp((char*)resp_ds.getbuffer(), resp_ds.size());
        protocol::MsgHeader msg_head;
        protocol::ExchangeKeyResponse msg_key_resp;
        recvds_resp >> msg_head;
        recvds_resp >> msg_key_resp;
        ret = client.OnExchangeKeyResponse(msg_key_resp);
        assert(ret>=0);
    }
    
    //5. client encry msg and send!
    char szSendBuff[512];
    CDataStream sendds(szSendBuff, sizeof(szSendBuff));
    {
        string msg_body = "AAAAAABBBBBB1234567890";
        protocol::MsgHeader msg_header;
        msg_header.msg_size = sizeof(protocol::MsgHeader)+msg_body.length()+1;
        msg_header.msg_id = 0xF5F5F5F5;
        msg_header.msg_type = 0x00FFFFFF;
        msg_header.msg_seq = 0xEE00EEEE;
        sendds << msg_header;
        sendds.writestring(msg_body.c_str());

        ret = client.EncryMsg((uint8*)sendds.getbuffer(), sendds.size());
        assert(ret>=0);
        //assert(strcmp(szMsg, "12345678901234567890")!=0);
        printf("encry--->msg_header(len:%u, id:0x%x, type:0x%x, seq:0x%x), msg_body(%s)\n",
            msg_header.msg_size,
            msg_header.msg_id,
            msg_header.msg_type,
            msg_header.msg_seq,
            msg_body.c_str());
    }

    //6. server receive msg and decry!
    {
        CDataStream recvds((char*)sendds.getbuffer(), sendds.size());

        ret = server.DecryMsg((uint8*)recvds.getbuffer(), recvds.size());
        assert(ret>=0);

        protocol::MsgHeader msg_header;
        string msg_body;
        
        recvds >> msg_header;
        msg_body = recvds.readstring();
        printf("decry--->msg_header(len:%u, id:0x%x, type:0x%x, seq:0x%x), msg_body(%s)\n",
             msg_header.msg_size,
             msg_header.msg_id,
             msg_header.msg_type,
             msg_header.msg_seq,
             msg_body.c_str());
    }    

    return 0;
}