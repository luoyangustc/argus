#ifdef _WINDOWS
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <list>
#include "datastream.h"
#include "AYClient.h"

using namespace std;
using namespace boost::asio;

//ip::tcp::endpoint serv_ep( ip::address::from_string("192.168.16.111"), 8000);
//ip::tcp::endpoint serv_ep( ip::address::from_string("139.196.230.124"), 9000);
//ip::tcp::endpoint serv_ep( ip::address::from_string("192.168.18.183"), 8000);
//ip::tcp::endpoint serv_ep( ip::address::from_string("192.168.16.105"), 8000);
ip::tcp::endpoint serv_ep( ip::address::from_string("139.196.255.159"), 8000);
io_service service;
uint32 gAyClientId = 0;
list<CAYClient_ptr> gAyClientList;
boost::recursive_mutex gLock;

void create_client(int client_num)
{
    while(client_num)
    {
        CAYClient_ptr spClient = CAYClient_ptr( new  CAYClient() );
        if( !spClient->Start(inet_addr(serv_ep.address().to_string().c_str()), serv_ep.port()) )
        {
            printf("client(%p) start failed\n", spClient.get());
            boost::this_thread::sleep( boost::posix_time::millisec(100));
            continue;
        }

        int wait = 20;
        while(!spClient->IsConnected() && wait--)
        {
            boost::this_thread::sleep( boost::posix_time::millisec(50));
        }

        wait = 40;
        while(!spClient->IsConnected() && wait--)
        {
            boost::this_thread::sleep( boost::posix_time::millisec(100));
        }

        if( !spClient->IsConnected() )
        {
            printf("client(%p) connect failed\n", spClient.get());
            continue;
        }

        client_num --;
        {
            boost::lock_guard<boost::recursive_mutex> lock(gLock);
            spClient->SetClientId(++gAyClientId);
            gAyClientList.push_back(spClient);
        }
    }
}

void client_heartbeat()
{
    uint32 heartbeat_cnt = 0;
    while(1)
    {
        list<CAYClient_ptr> client_list;
        {
            boost::lock_guard<boost::recursive_mutex> lock(gLock);
            client_list = gAyClientList;
        }

        if( client_list.empty() )
        {
            boost::this_thread::sleep( boost::posix_time::millisec(5*1000));
            continue;
        }

        ++heartbeat_cnt;

        printf("client_heartbeat-->Start, ClientNum(%u), heartbeat(%u)=== \n", client_list.size(), heartbeat_cnt);

        list<CAYClient_ptr>::iterator itor = client_list.begin();
        while(itor != client_list.end())
        {
            CAYClient_ptr pClient = *itor;
            pClient->Heartbeat(heartbeat_cnt);
            ++itor;
        }

        printf("client_heartbeat-->End=======================================\n");

        boost::this_thread::sleep( boost::posix_time::millisec(10*1000));
    }
    
}

void run(int thread_idx)
{
    service.run();
}

int main(int argc, char* argv[]) 
{
    uint32 total_client_num = 500;

    if( argc == 2 )
    {
        total_client_num = (uint32)atoi(argv[1]);
        if( total_client_num == 0 )
        {
            total_client_num = 500;
        }
    }
    else if( argc == 4 )
    {
        string serv_ip = argv[1];
        uint16 serv_port = atoi(argv[2]);
        if( serv_port == 0 )
        {
            serv_port = 8000;
        }
        ip::tcp::endpoint ep( ip::address::from_string(serv_ip.c_str()), serv_port);
        serv_ep = ep;

        total_client_num = (uint16)atoi(argv[3]);
        if( total_client_num == 0 )
        {
            total_client_num = 500;
        }
    }
    
    boost::asio::io_service::work service_work(service);

    boost::thread_group iosvc_threads;
    for ( int i = 0; i<8; i++)
    {
        iosvc_threads.create_thread( boost::bind(run, i));
    }

    boost::thread_group work_threads;
    int THD_NUM = 20;
    for( int i=1; i<=THD_NUM; i++)
    {
        uint32 client_num = total_client_num/THD_NUM;
        if(i==THD_NUM)
        {
            client_num += total_client_num%THD_NUM;
        }
        work_threads.create_thread( boost::bind(create_client, client_num));
    }
    work_threads.create_thread( boost::bind(client_heartbeat));
    
    iosvc_threads.join_all();
    work_threads.join_all();

    boost::this_thread::sleep( boost::posix_time::millisec(30*1000));
    //sock.close();
}