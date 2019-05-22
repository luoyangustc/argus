#ifndef __UDP_SERVER_H_
#define __UDP_SERVER_H_

#include <map>
#include <queue>
#include "Common.h"
#include "IServerLogical.h"
#include "typedefine.h"
#include "ServerBase.h"
#include "HandlerAllocator.h"
#include "HostInfo.h"
#include "BufferInfo.h"

using namespace std;
using namespace boost::asio;

struct SUDPMessageInfo
{
    CHostInfo hi_remote;
    SDataBuff data;
};

class CUDPServer:public IServerBase, public boost::enable_shared_from_this<CUDPServer>  
{
public:
    CUDPServer( boost::asio::io_service& io_service, ip::udp::endpoint& serv_ep);
    virtual ~CUDPServer();
    int Init( IServerLogical* serv_logic );
    int Start();
    int Stop();
    void Update();
    string GetServerName();
    std::ostringstream& DumpInfo(std::ostringstream& oss);
private:
    bool WorkRunProc(uint32 thread_idx);
    bool ReadRunProc();
    bool StartAsyncRead();
    void HandleRead(const boost::system::error_code& error, std::size_t read_bytes);

    bool PushQueue(IN const SUDPMessageInfo& msg_info);
    bool PopQueue(OUT SUDPMessageInfo& msg_info);
    
private:
    bool                                m_bRunning;
    IServerLogical*                     m_pServiceLogic;
    boost::asio::io_service&            m_IoService;
    boost::asio::ip::udp::endpoint      m_ServEndpoint;
    boost::asio::ip::udp::socket        m_ServSocket;
    handler_allocator			        m_Allocator;
    char                                m_RecvBuffer[8*1024];
    std::queue<SUDPMessageInfo>         m_WorkQueue;
    boost::mutex                        m_WorkMutex;
    boost::condition                    m_WorkCond;
    boost::mutex                        m_ReadMutex;
    boost::condition                    m_ReadCond;
    uint32                              m_unWorkThreadCnt;
    boost::thread_group                 m_WorkThreadGroup;
    boost::shared_ptr<boost::thread>    m_ReadThread;

    //boost::asio::deadline_timer m_Timer;	// The timer for repeat accept delay.
};

#endif