
#include "ServerLogical.h"
#include "Server/ServerApi.h"

CServerLogical CServerLogical::logic_;

CServerLogical::CServerLogical()
{
}

CServerLogical::~CServerLogical()
{


}

bool CServerLogical::Start()
{
	start_tick_ = get_current_tick();
	
	return true;
}

void CServerLogical::Stop()
{

}

void CServerLogical::Update()
{

}

void CServerLogical::DoIdleTask()
{
    
}

//消息类型定义
enum MSG_TYPE
{
    MSG_TYPE_EXCHANGE_REQ   = 0x00000001,   //密钥交换请求
    MSG_TYPE_EXCHANGE_RESP  = 0x00000002,   //密钥交换响应
    MSG_TYPE_CMD_REQ        = 0x00000003,   //Command请求
    MSG_TYPE_CMD_RESP       = 0x00000004,   //Command响应
    MSG_TYPE_ECHO_TEST      = 0x00000005,   //反射测试消息
};

int32 CServerLogical::OnTCPMessage(ITCPSessionSendSink*sink,CHostInfo& hiRemote,uint32 msg_type, CDataStream& recvds,CDataStream& sendds)
{
    printf("(%s,%u),msg_type:(%x),from:(%s),size(%u)\n",__FUNCTION__,__LINE__, msg_type,hiRemote.GetNodeString().c_str(),recvds.getbuffer_length());
    do
    {
        switch(msg_type)
        {
        case MSG_TYPE_ECHO_TEST:
            {
                string msg_body;
                recvds >> msg_body;
                printf( "(%s,%u)echo msg-->%s!\n",__FUNCTION__,__LINE__, msg_body.c_str() );
                
                uint16 msg_size = sizeof(uint16) + sizeof(uint32) + recvds.getbuffer_length();
                uint32 msg_type = MSG_TYPE_ECHO_TEST;
                sendds << msg_size;
                sendds << msg_type;
                sendds << msg_body;
            }
            break;
        default:
            {
                printf("(%s,%u)invalid message(%x,%x,%u)!\n",__FUNCTION__,__LINE__, msg_type,hiRemote.IP,hiRemote.Port);
            }
            break;
        };

        return 0;

    }while(false);

    return -1;
}

int32 CServerLogical::OnTCPAccepted(ITCPSessionSendSink*sink,CHostInfo& hiRemote,CDataStream& sendds)
{
	do
	{
		printf("(%s,%u)message(%s)!\n",__FUNCTION__,__LINE__, hiRemote.GetNodeString().c_str());

        return 0;

	}while(false);

	return -1;
}

int32 CServerLogical::OnTCPClosed(ITCPSessionSendSink*sink,CHostInfo& hiRemote)
{
	do
	{
		printf("(%s,%u)closed(%x,%u)!\n",__FUNCTION__,__LINE__, hiRemote.IP,hiRemote.Port);

        return 0;

	}while(false);

	return -1;
}

int32 CServerLogical::OnUDPMessage(CHostInfo& hiRemote, CDataStream& recvds, CDataStream& sendds, IN int thread_index,uint8 algo)
{
	return 0;
}

int32 CServerLogical::OnHttpClientRequest(CHostInfo& hiRemote,SHttpRequestPara_ptr pReq,SHttpResponsePara_ptr pRes)
{
    //printf("(%s,%u),from:(%s)\n",__FUNCTION__,__LINE__, hiRemote.GetNodeString().c_str());
    do
    {
        if( ! pReq->header_ptr->is_request )
        {
            printf("(%s,%u)Parse Message Error!\n",__FUNCTION__,__LINE__);
            break;
        }

        const string STATUS_FILE_NAME = "/status_detail.json";

        if (pReq->header_ptr->url_ == "/")
        {
            static string s_content = "<html><title>result</title><body>Test Serv!</body></html>";
            static int s_content_length = s_content.length();

            pRes->pContent = boost::shared_array<uint8>(new uint8[s_content_length+1]);
            memcpy(pRes->pContent.get(),s_content.c_str(),s_content_length+1);
            pRes->content_len = s_content.length();
            pRes->ret_code = "200 OK";
            pRes->content_type = "text/html";
        } 
        else if (pReq->header_ptr->url_.substr(0,STATUS_FILE_NAME.length()) == STATUS_FILE_NAME)
        {
            string type;
            map<string,string>::iterator it = pReq->header_ptr->url_detail_.params_.find("type");
            if (it != pReq->header_ptr->url_detail_.params_.end())
            {
                type = it->second;
            }
            
            if(type == "core")
            {
                ostringstream oss;
                DumpInfo(oss, type);
                pRes->content_len = oss.str().length();
                pRes->pContent = boost::shared_array<uint8>(new uint8[pRes->content_len]);
                memcpy(pRes->pContent.get(), oss.str().c_str(), pRes->content_len);
                pRes->ret_code = "200 OK";
                pRes->content_type = "application/json";
            }
            else
            {
                pRes->ret_code = "404 Not Found";
                pRes->keep_alive = false;
            }
        }

#if 0
        string strReqest="";
        if( pReq->content_len )
        {
            strReqest.assign((const char*)pReq->content_ptr.get(), pReq->content_len);
            printf("(%s,%u),from:(%s), request_content(%s)\n",__FUNCTION__,__LINE__, hiRemote.GetNodeString().c_str(), strReqest.c_str());
        }

        string strPrefix = "echo " ;
        string strResponse = strPrefix + strReqest;
        printf("(%s,%u),from:(%s), request_content(%s)\n",__FUNCTION__,__LINE__, hiRemote.GetNodeString().c_str(), strResponse.c_str());

        pRes->content_len = strResponse.length();
        pRes->pContent = boost::shared_array<uint8>(new uint8[pRes->content_len+1]);        
        memcpy(pRes->pContent.get(), strResponse.data(), strResponse.length());       
        
        pRes->ret_code = "200 OK";
        pRes->content_type = "text/octet-stream";
#endif
        return 0;

    } while(false);

    return -1;
}

ostringstream& CServerLogical::DumpInfo(ostringstream& oss,const string&type)
{
    oss << "{" ;
    {
        oss << "\"type\":\"";
        oss << "test_serv";
        oss << "\"";
        if (type == "core")
        {
            oss << ",";
            oss << "\"core\":";
            Serv_Core_Dump(oss);
        }
    }
    oss << "}" ;
    
	return oss;
}



