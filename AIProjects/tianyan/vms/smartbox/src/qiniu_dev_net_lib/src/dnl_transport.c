#define __DNL_TRANSPORT_C__

#include "vos_socket.h"
#include "vos_sock.h"
#include "vos_bit_t.h"
#include "cdh_crypt_lib.h"
#include "cdiffie_hell_man.h"
#include "dnl_transport.h"
#include "dnl_dev.h"
#include "dnl_log.h"
#include "dnl_ctrl.h"
#include "dnl_entry.h"
#include "dnl_stream.h"

static int dnl_build_msg_exchange_req(void *msg_buf, vos_size_t buf_len, vos_uint32_t msg_seq, vos_sock_t fd);
static int dnl_on_msg_exchange_resp(vos_uint8_t *data_buf, vos_uint32_t data_len, vos_sock_t fd);
static int dnl_build_msg_resp(OUT void *msg_buf, IN vos_size_t buf_len, IN vos_uint32_t msg_id, IN vos_uint32_t msg_seq, IN void* resp);

static vos_bool_t is_blocking()
{
    int err = vos_get_native_netos_error();
    if( ( err == VOS_EWOULDBLOCK)
    || ( err == VOS_EINTR ) ) 
    {
        return TRUE;
    }

    return FALSE;
}

int dnl_transport_open(dnl_transport_cfg_t* cfg, dnl_transport_data_t* tp)
{
    vos_sockaddr_in rmt_addr;
    vos_uint32_t value;
    vos_in_addr* peer_ip;
    vos_int32_t error_code;
    
    if( (!cfg) || (!tp) ) 
    {
        return -1;
    }
    
    dnl_transport_rtx_clear(tp);

    //上层没有设置tx,rx的buffer长度，此处设置默认长度
    {
        if ( tp->tx.buf == NULL )
        {
            tp->tx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, TP_TX_DEF_BUFF_LEN);
            tp->tx.max_size = TP_TX_DEF_BUFF_LEN;
        }

        if ( tp->rx.buf == NULL )
        {
            tp->rx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, TP_RX_DEF_BUFF_LEN);
            tp->rx.max_size = TP_RX_DEF_BUFF_LEN;
        }
    } 

    tp->sock_type = cfg->sock_type;
    tp->peer_host.sin_addr = cfg->peer_host.sin_addr;
    tp->peer_host.sin_port = cfg->peer_host.sin_port;

    tp->sock = socket(AF_INET, tp->sock_type,  0);
    if (tp->sock == VOS_INVALID_SOCKET)
    {
        DNL_ERROR_LOG(
                "(%s:%d, %s), get socket failed, errno=%d!\n", 
                vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
                vos_ntohs(tp->peer_host.sin_port), 
                (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
                vos_get_native_netos_error() );
        goto on_error_exit;
    }

    DNL_TRACE_LOG("tp(%p), sock_fd(%d)!\n", tp, tp->sock);

    if(cfg->is_block)
    {
        /* 设置非阻塞 */
        value = 1;
        #if (OS_LINUX == 1)
        if (ioctl(tp->sock, FIONBIO, &value)) 
        #else
        if (ioctlsocket(tp->sock, FIONBIO, &value)) 
        #endif
        {
            DNL_ERROR_LOG(
                    "tp(%p), set socket ioctl failed, sock_fd(%d), errno(%d)!\n", 
                    tp, 
                    tp->sock, 
                    vos_get_native_netos_error() );
            goto on_error_exit;
        }
    }
    
    if(tp->sock_type == SOCK_STREAM)
    {
        memset(&rmt_addr, 0x0, sizeof(rmt_addr));
        rmt_addr.sin_family = AF_INET;
        rmt_addr.sin_addr.s_addr = tp->peer_host.sin_addr;
        rmt_addr.sin_port = tp->peer_host.sin_port;

        //struct sockaddr
        peer_ip = (vos_in_addr*)&rmt_addr.sin_addr.s_addr;
        
        if(connect(tp->sock, (struct sockaddr*)&rmt_addr, sizeof(rmt_addr)) < 0)
        {
            error_code = vos_get_native_netos_error();	    
            DNL_ERROR_LOG(
                    "tp(%p), connect to %s:%d, sock_fd(%d), errno(%d)!\n", 
                    tp, 
                    vos_inet_ntoa(*peer_ip),
                    vos_ntohs(rmt_addr.sin_port),
                    tp->sock,
                    error_code);
                    
            if ( error_code == VOS_EINPROGRESS )
            {
                tp->tcp_state = en_tcp_state_connecting;
            }
            else
            {
                goto on_error_exit;
            }
        }
        else
        {
            tp->tcp_state = en_tcp_state_connected;
        }
    }

    DNL_DEBUG_LOG(
            "tp(%p),(%s:%d, %s),sock_fd(%d), open success!\n",
            tp, 
            vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
            vos_ntohs(tp->peer_host.sin_port), 
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
            tp->sock);
    return VOS_SUCCESS;
	
on_error_exit:
    if (tp->sock != VOS_INVALID_SOCKET)
    {
        vos_sock_close(tp->sock);
        tp->sock = VOS_INVALID_SOCKET;
        tp->tcp_state = en_tcp_state_none;
    }
    DNL_ERROR_LOG(
            "tp(%p),(%s:%d, %s),sock_fd(%d), open failed!\n", 
            tp, 
            vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
            vos_ntohs(tp->peer_host.sin_port), 
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
            tp->sock );
    
    return -1;
}

int dnl_transport_tcp_connect_check(dnl_transport_data_t* tp)
{
    int ret = -1;
    fd_set  rd_fds, wr_fds;
    struct timeval t_timeout = {0, 2*1000};
    
    FD_ZERO(&rd_fds);
    FD_SET(tp->sock, &rd_fds);
    wr_fds = rd_fds;

    ret = select(tp->sock + 1, &rd_fds, &wr_fds, NULL, &t_timeout);
    if(ret < 0)
    {
        DNL_ERROR_LOG(
            "tp(%p),(%s:%d, %s),sock_fd(%d),connect failed, errno=%d!\n", 
            tp,
            vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
            vos_ntohs(tp->peer_host.sin_port), 
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
            tp->sock, 
            vos_get_native_netos_error() );
        return -1;
    }
    else if(ret > 0)
    {
        DNL_ERROR_LOG(
            "tp(%p),(%s:%d, %s),sock_fd(%d),already connected!\n", 
            tp,
            vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
            vos_ntohs(tp->peer_host.sin_port), 
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
            tp->sock);
                
        if( FD_ISSET(tp->sock, &rd_fds) || FD_ISSET(tp->sock, &wr_fds))
        {
            tp->tcp_state = en_tcp_state_connected;
            return VOS_SUCCESS;
        }
    }
    
    return -1;
}

vos_status_t dnl_transport_send(dnl_transport_data_t* tp)
{
	vos_sockaddr_in rmt_addr;
	vos_in_addr* peer_ip;
    //struct timeval t_timeout = {0, 50*1000};
    
	//int i=0;

    if( tp->tx.sent_size > tp->tx.data_size )
    {
        tp->tx.flag = en_tp_flag_err;

        DNL_ERROR_LOG(
            "tp(%p),(%s:%d, %s),sock_fd(%d),sent_size(%d) > data_size(%d),error!\n",
            tp,
            vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr),
            vos_ntohs(tp->peer_host.sin_port),
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
            tp->sock,
            tp->tx.sent_size,
            tp->tx.data_size );
        return -1;
    }

    {
        int sent = 0;
        vos_uint8_t* send_data_buf = &tp->tx.buf[tp->tx.sent_size];
        vos_size_t send_data_size = tp->tx.data_size - tp->tx.sent_size;

    	if(tp->sock_type == SOCK_DGRAM)
    	{
    	    rmt_addr.sin_family = AF_INET;
        	rmt_addr.sin_addr.s_addr = tp->peer_host.sin_addr;
        	rmt_addr.sin_port = tp->peer_host.sin_port;

        	peer_ip = (vos_in_addr*)&rmt_addr.sin_addr.s_addr;
        	
    		sent = sendto(tp->sock, (const char*)send_data_buf, send_data_size, 0, (struct sockaddr*)&rmt_addr, sizeof(rmt_addr) );
    	}
    	else
    	{
    		sent = send(tp->sock, (const char*)send_data_buf, send_data_size, 0 );
    	}
        
    	if ( sent < 0 )
    	{
            DNL_ERROR_LOG(
                "tp(%p),(%s:%d, %s),sock_fd(%d),data_size(%d), sent_size(%d), ret=%d!\n",
                tp,
                vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr),
                vos_ntohs(tp->peer_host.sin_port),
                (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
                tp->sock,
                tp->tx.data_size,
                tp->tx.sent_size,
                sent );
            if ( is_blocking() )
    	    {
                tp->tx.flag = en_tp_flag_writing;
                return 0;
    	    }
    	    
    	    tp->tx.flag = en_tp_flag_err;
    	    if(tp->sock_type == SOCK_STREAM)
    	    {
    	        tp->tcp_state = en_tcp_state_disconnect;
    	    }
            return -1;
    	}
    	else 
    	{
    	    tp->tx.sent_size += sent;

            if( tp->tx.sent_size >= tp->tx.data_size)
            {
                tp->tx.flag = en_tp_flag_complete;
            }
            else//已发送部分数据
            {
                tp->tx.flag = en_tp_flag_writing;
            }
    	}
    }
    
    return 0;
}

int dnl_transport_recv(dnl_transport_data_t* tp)
{
    vos_ssize_t len = 0;
    vos_sockaddr_in rmt_addr;
    vos_in_addr* peer_ip;
    //vos_sock_t fd_max = 0;
    fd_set  select_fds;
    struct timeval t_timeout = {0, 0};
    int ret = 0;

    FD_ZERO(&select_fds);
    FD_SET(tp->sock, &select_fds);

    ret = select(tp->sock + 1, &select_fds, NULL, NULL, &t_timeout);
    
    if(ret>0 && FD_ISSET(tp->sock, &select_fds))
    {
        char* recv_buf = (char*)tp->rx.buf + tp->rx.data_size;
        int recv_buf_len = tp->rx.max_size - tp->rx.data_size;

    	if(tp->sock_type == SOCK_DGRAM)
    	{
            vos_socklen_t addr_len;
    	    rmt_addr.sin_family = AF_INET;
        	rmt_addr.sin_addr.s_addr = tp->peer_host.sin_addr;
        	rmt_addr.sin_port = tp->peer_host.sin_port;
        	
    		addr_len = sizeof(rmt_addr);
    		len = recvfrom(tp->sock, recv_buf, recv_buf_len, 0, (struct sockaddr*)&rmt_addr, (socklen_t*)&addr_len);
    	}
    	else
    	{
    		len = recv(tp->sock, recv_buf, recv_buf_len, 0);
    	}

        peer_ip = (vos_in_addr*)&tp->peer_host.sin_addr;

        DNL_TRACE_LOG(
            "tp(%p), (%s:%d, %s),sock_fd(%d),recv_len(%d), errno=%d.\n", 
            tp,
    		vos_inet_ntoa(*peer_ip), 
    		vos_ntohs(tp->peer_host.sin_port),
            (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
    		tp->sock, 
            len, 
            vos_get_native_netos_error() );
    		
    	if(len>0)
    	{
            tp->rx.data_size += len;
            tp->rx.flag = en_tp_flag_reading;
    	}
        else if (len==0)
        {
            if((tp->sock_type == SOCK_STREAM))
            {
                tp->tcp_state = en_tcp_state_disconnect;
                DNL_ERROR_LOG(
                    "tp(%p), (%s:%d, %s),sock_fd(%d), disconnect, ret(%d), errno(%d).\n", 
                    tp,
                    vos_inet_ntoa(*peer_ip), 
                    vos_ntohs(tp->peer_host.sin_port),
                    (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
                    tp->sock, 
                    len, 
                    vos_get_native_netos_error() );
                return -1;
            }
        }
        else
        {
            if ( !is_blocking() )
    	    {
                tp->tcp_state = en_tcp_state_disconnect;
                
                DNL_ERROR_LOG(
                    "tp(%p), (%s:%d, %s),sock_fd(%d), recv error, ret(%d), errno(%d).\n", 
                    tp,
                    vos_inet_ntoa(*peer_ip), 
                    vos_ntohs(tp->peer_host.sin_port),
                    (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
                    tp->sock, 
                    len, 
                    vos_get_native_netos_error() );

                return -1;
    	    }	    
        }
    }
	return len;
}

int dnl_transport_close(dnl_transport_data_t* tp)
{
    tp->tcp_state = en_tcp_state_close;
    
    DNL_WARN_LOG(
        "tp(%p), (%s:%d, %s),sock_fd(%d), close.\n", 
        tp,
        vos_inet_ntoa(*(vos_in_addr*)&tp->peer_host.sin_addr), 
        vos_ntohs(tp->peer_host.sin_port),
        (tp->sock_type == SOCK_STREAM)?"TCP":"UDP",
        tp->sock );

    dnl_transport_rtx_clear(tp);

    if( tp->sock != VOS_INVALID_SOCKET )
    {
        vos_sock_close(tp->sock);
        tp->sock = VOS_INVALID_SOCKET;
    }

    return 0;
}

tp_tcp_state_e dnl_transport_tcp_state( dnl_transport_data_t* tp )
{
    return tp->tcp_state;
}

vos_bool_t dnl_transport_tx_writing( dnl_transport_data_t* tp )
{
    if( (tp->sock_type == SOCK_STREAM) && (tp->tcp_state != en_tcp_state_connected) )
    {
        return FALSE;
    }

    if( tp->tx.flag == en_tp_flag_writing )
    {
        return TRUE;
    }

    return FALSE;
}

vos_bool_t dnl_transport_tx_enable( dnl_transport_data_t* tp )
{
    if( (tp->sock_type == SOCK_STREAM) && (tp->tcp_state != en_tcp_state_connected) )
    {
        return FALSE;
    }

    if ( ( tp->tx.flag == en_tp_flag_none ) || ( tp->tx.flag == en_tp_flag_complete ) )
    {
        return TRUE;
    }

    return FALSE;
}

vos_bool_t dnl_transport_rx_enable( dnl_transport_data_t* tp )
{
    if ( !tp )
    {
        return FALSE;
    }

    if ( tp->rx.flag != en_tp_flag_none )
    {
        return FALSE;
    }

    return TRUE;
}

vos_bool_t dnl_transport_rx_complete( dnl_transport_data_t* tp )
{
    if ( tp->rx.flag != en_tp_flag_complete )
    {
        return FALSE;
    }

    return TRUE;
}

void dnl_transport_tx_clear(dnl_transport_data_t* tp)
{
	tp->tx.data_size = 0;
	tp->tx.sent_size = 0;
    tp->tx.flag = en_tp_flag_none;
}

void dnl_transport_rx_clear(dnl_transport_data_t* tp)
{
	tp->rx.flag = en_tp_flag_none;
	tp->rx.data_size = 0;
}

void dnl_transport_rtx_clear(dnl_transport_data_t* tp)
{
	dnl_transport_tx_clear(tp);
	dnl_transport_rx_clear(tp);
}

int dnl_tp_tx_data(dnl_transport_data_t* tp, int msg_id, int msg_type, void* arg1, void* arg2)
{
	int ret = -1;
	vos_sock_t fd;
	int key_len;
	char key_buf[64];
	//int data_len = 0;

	if ( !tp )
	{
        DNL_ERROR_LOG("dnl_tp_tx_data--->tp is NULL\n");
		return -1;
	}

    fd = tp->sock;
	
    if(msg_type==MSG_TYPE_REQ)
    {
        switch(msg_id)
	    {
	    case MSG_ID_EXCHANGE_KEY:
            {
                ret = dnl_build_msg_exchange_req( tp->tx.buf, tp->tx.max_size, *(vos_uint32_t*)arg1, fd );
            }
		    break;
        
	    case MSG_ID_DEV_LOGIN:
            {
                ret = entry_build_msg_device_login_req(tp->tx.buf, tp->tx.max_size);
            }
		    break;
        case MSG_ID_DEV_ABILITY_REPORT:
            {
                ret = entry_build_msg_ability_report_req(tp->tx.buf, tp->tx.max_size);
            }
            break;
	    case MSG_ID_DEV_STATUS_REPORT:
            {
                ret = entry_build_msg_status_report_req(tp->tx.buf, tp->tx.max_size);
            }
		    break;
        case MSG_ID_DEV_ALARM_REPORT:
            {
                ret = entry_build_msg_alarm_report_req(tp->tx.buf, tp->tx.max_size);
            }
            break;
        case MSG_ID_MEDIA_CONNECT:
            {
                ret = stream_build_msg_media_connect_req((media_session_t*)arg1, tp->tx.buf, tp->tx.max_size);
            }
            break;
        case MSG_ID_MEDIA_DISCONNECT:
            {
                ret = stream_build_msg_media_disconnect_req((media_session_t*)arg1, tp->tx.buf, tp->tx.max_size);
            }
            break;
        case MSG_ID_MEDIA_STATUS:
            {
                ret = stream_build_msg_media_status_report_req((media_session_t*)arg1, tp->tx.buf, tp->tx.max_size);
            }
            break;
	    default:
		    break;
	    }
    }
    else if(msg_type == MSG_TYPE_RESP)
    {
        switch(msg_id)
	    {
        case MSG_ID_DEV_MEDIA_OPEN:
        case MSG_ID_DEV_MEDIA_CLOSE:
        case MSG_ID_DEV_SNAP:
        case MSG_ID_DEV_CTRL:
        case MSG_ID_MEDIA_PLAY:
        case MSG_ID_MEDIA_PAUSE:
        case MSG_ID_MEDIA_CMD:
        case MSG_ID_MEDIA_CLOSE:
            {
                ret = dnl_build_msg_resp(tp->tx.buf, tp->tx.max_size, msg_id, *(vos_uint32_t*)arg1, arg2);
            }
            break;
	    default:
		    break;
	    }
    }
    else if(msg_type == MSG_TYPE_NOTIFY)
    {
        switch(msg_id)
	    {
        case MSG_ID_MEDIA_FRAME:
            {
                //视频帧通知消息已经在tp->tx的buffer中
                ret = tp->tx.data_size;
            }
            break;
        case MSG_ID_MEDIA_EOS:
            {
                ret = stream_build_msg_media_eos_notify((media_session_t*)arg1, tp->tx.buf, tp->tx.max_size);
            }
            break;
	    default:
		    break;
	    }
    }
	
	if( ret < 0)
	{
	    DNL_ERROR_LOG("msg_id(0x%x), msg_type(0x%x), fd(%d), ret(%d).\n", msg_id, msg_type, tp->sock, ret);
		return ret;
	}

    /* S--> set send info...*/
	tp->tx.data_size =  ret;
    tp->tx.sent_size = 0;
    tp->tx.flag = en_tp_flag_writing;
    /* E--> set send info...*/

	if ( tp->sock_type == SOCK_DGRAM )
	{
		ay_udp_msg_encrypt( tp->tx.buf, tp->tx.data_size );
	}
	else if ( tp->sock_type == SOCK_STREAM )
	{
		if ( STREAM_ENCRYPT && (msg_id != MSG_ID_EXCHANGE_KEY) )
		{
			fd = tp->sock;
			key_len = Get_exchangekey_( fd, key_buf );
			if ( key_len <= 0 )
			{
				DNL_DEBUG_LOG("unpack get key err\n");
				return -1;
			}

			if ( ay_tcp_msg_encrypt( tp->tx.buf, tp->tx.data_size, key_buf, key_len) != 0 )
			{
				return -1;
			}
		}
	}
	else
	{
	    DNL_ERROR_LOG("sock_type(%d) error, fd = %d.\n",tp->sock_type, tp->sock);
		return -1;
	}


	return dnl_transport_send( tp );
}

int dnl_tp_rx_data(dnl_transport_data_t* tp, vos_uint32_t msg_id, vos_uint32_t msg_type, vos_uint32_t msg_seq)
{
    int ret = 0, left_data_len = 0;
    char* body_pos = NULL;
    vos_size_t body_len = 0;
    MsgHeader header;
    
    ret = dnl_transport_recv(tp);
    if(ret < 0)
    {
        return en_tp_rx_err;
    }
    else
    {
        //检查消息头是否接受完整
        if ( tp->rx.data_size < sizeof(MsgHeader) )
        {
            return en_tp_rx_recving;
        }

        //检查消息是否接受完整
        {
            vos_uint16_t msg_size;
            char* pos = (char*)tp->rx.buf;
            R2Bytes(pos, msg_size);
            if ( tp->rx.data_size < msg_size )
            {
                return en_tp_rx_recving;
            }
        }
    }

    if ( tp->sock_type == SOCK_DGRAM )
    {
        ay_udp_msg_decrypt(tp->rx.buf, tp->rx.data_size);
    }
    else if ( tp->sock_type == SOCK_STREAM )
    {
        vos_uint16_t msg_size;
        vos_uint32_t msg_id;
        char* pos = (char*)tp->rx.buf;
        R2Bytes(pos, msg_size);
        R4Bytes(pos, msg_id);

        if ( STREAM_ENCRYPT && (msg_id != MSG_ID_EXCHANGE_KEY ) )
        {
            int key_len;
            char key_buf[64];
            key_len = Get_exchangekey_(tp->sock, key_buf);
            if ( key_len <= 0 )
            {
                DNL_DEBUG_LOG("unpack get key err\n");  
                return en_tp_rx_err;
            }
			
            if( ay_tcp_msg_decrypt(tp->rx.buf, tp->rx.data_size, key_buf, key_len) != 0 )
            {
                DNL_DEBUG_LOG("decry_1 err\n");
                return en_tp_rx_err;
            }
        }
    }

    if ( Unpack_MsgHeader(tp->rx.buf, sizeof(MsgHeader), &header) < 0 )
    {
        return en_tp_rx_err;
    }

    /*msg_id为指定的消息解析，如果为0表示任何消息都解析*/
    if(  msg_id && 
         ( (header.msg_id != msg_id ) || 
           (header.msg_type!=msg_type) || 
           (header.msg_seq!=msg_seq) ) )
    {
        return en_tp_rx_err;
    }

    body_pos = (char*)tp->rx.buf + sizeof(MsgHeader);
    body_len = tp->rx.data_size - sizeof(MsgHeader);

    switch(header.msg_id)
    {
    case MSG_ID_EXCHANGE_KEY:
        {
            if(header.msg_type==MSG_TYPE_RESP)
            {
                ret = dnl_on_msg_exchange_resp( tp->rx.buf, tp->rx.max_size, tp->sock );
            }
        }
        break;
    case MSG_ID_DEV_LOGIN:
        {
            if(header.msg_type!=MSG_TYPE_RESP)
            {
                return en_tp_rx_err;
            }
            else
            {
                DeviceLoginResp login_resp;
                if ( Unpack_MsgDeviceLoginResp( body_pos, body_len, &login_resp) < 0 )
                {
                    return en_tp_rx_err;
                }

                if( entry_on_msg_device_login_resp( &login_resp ) == RS_NG )
                {
                    return en_tp_rx_err;
                }
            }
        }
        break;
    case MSG_ID_DEV_ABILITY_REPORT:
    case MSG_ID_DEV_STATUS_REPORT:
    case MSG_ID_DEV_ALARM_REPORT:
    case MSG_ID_DEV_MEDIA_OPEN:
    case MSG_ID_DEV_MEDIA_CLOSE:
    case MSG_ID_DEV_SNAP:
    case MSG_ID_DEV_CTRL:
        {
            //会话服务相关的消息处理
            //ret = entry_on_recv_msg(tp, &header, body_pos, body_len);
            ret = entry_unpack_recv_msg(tp);
        }
        break;
    case MSG_ID_MEDIA_CONNECT:
    case MSG_ID_MEDIA_DISCONNECT:
    case MSG_ID_MEDIA_PLAY:
    case MSG_ID_MEDIA_PAUSE:
    case MSG_ID_MEDIA_CMD:
    case MSG_ID_MEDIA_CLOSE:
        {
            //流服务相关的消息处理
            ret = stream_on_recv_msg(tp, &header, body_pos, body_len);
        }
        break;
    default:
        break;
    }

    //检查接收消息的处理结果
    if(ret<0)
    {
        return en_tp_rx_err;
    }

    //将rx_buf中剩余的数据移动到rx_buf的头，方便下个周期继续接受解析
    left_data_len = tp->rx.data_size - header.msg_size;
    if ( left_data_len > 0 )
    {
        memmove(tp->rx.buf, tp->rx.buf+header.msg_size, left_data_len);
        tp->rx.data_size -= header.msg_size;
    }
    else
    {
        tp->rx.data_size = 0;
    }

    return en_tp_rx_complete;
}

static int dnl_build_msg_exchange_req(void *msg_buf, vos_size_t buf_len, vos_uint32_t msg_seq, vos_sock_t fd)
{
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    if ( !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    do
    {
        struct ExchangeKeyRequest KeyReq;
        {
            memset(&KeyReq,0,sizeof(struct ExchangeKeyRequest));

            KeyReq.mask = 0x01;
            KeyReq.mask |= 0x02;
            if(InitDiffieHellman(8, fd) == -1)
            {
                DNL_DEBUG_LOG("exchpack req 4\n");

                break;
            }
            if(MakePrime(fd) == -1)
            {
                DNL_DEBUG_LOG("exchpack req 3\n");

                break;
            }
            if(ComputesA(fd) == -1)
            {
                break;
            }
            Printf_a_p(&KeyReq, fd);

            if(STREAM_ENCRYPT)
            {
                KeyReq.except_algorithm = 1001;  //0：不用加密， 1001：加密算法
                KeyReq.algorithm_param = DEVICE_1_KEY_POS;
            }
            else
            {
                KeyReq.except_algorithm = 0;   //测试，未加密的码流
            }
        }

        body_size = Pack_MsgExchangeKeyRequest(body_pos, buf_len-sizeof(MsgHeader), &KeyReq);
        if( body_size <= 0 )
        {
            DNL_DEBUG_LOG("pack exchange req msg failed\n");
            break;
        }

        msg_size = body_size + sizeof(MsgHeader);

        {
            MsgHeader header;
            header.msg_size = msg_size;
            header.msg_id = MSG_ID_EXCHANGE_KEY;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = msg_seq;
            if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
            {
                break;
            }
        }
        return msg_size;
    }while (0);
    return -1;
}

static int dnl_on_msg_exchange_resp(vos_uint8_t *data_buf, vos_uint32_t data_len, vos_sock_t fd)
{
    do 
    {
        ExchangeKeyValue exchg_key;
        ExchangeKeyResponse	exchg_rsp;

        memset(&exchg_rsp, 0, sizeof(exchg_rsp));

        if( Unpack_MsgExchangeKeyResponse((char*)data_buf, data_len, &exchg_rsp) != 0)
        {
            return -1;
        }

        if(exchg_rsp.keyB_01.key_size > sizeof(DWORD)*sizeof(exchg_key.key_size))
        {
            break;
        }

        Set_B(&exchg_rsp, fd);
        if(ComputesS1(fd) == -1)
        {
            break;
        }

        exchg_key.key_size = exchg_rsp.keyB_01.key_size;

        return Printf_S1(fd, &exchg_key.Key[0], sizeof(DWORD)*sizeof(exchg_key.key_size));

    } while (0);
    return -1;
}

static int dnl_build_msg_resp(OUT void *msg_buf, IN vos_size_t buf_len, IN vos_uint32_t msg_id, IN vos_uint32_t msg_seq, IN void* resp)
{
    int ret = 0;
    int msg_size;
    char *head_pos, *body_pos;

    if ( !msg_buf || !resp || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }
    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    switch(msg_id)
    {
    case MSG_ID_MEDIA_PLAY:
        {
            ret = Pack_MsgStreamMediaPlayResp(body_pos, buf_len-sizeof(MsgHeader), (StreamMediaPlayResp*)resp);
        }
        break;
    case MSG_ID_MEDIA_PAUSE:
        {
            ret = Pack_MsgStreamMediaPauseResp(body_pos, buf_len-sizeof(MsgHeader), (StreamMediaPauseResp*)resp);
        }
        break;
    case MSG_ID_MEDIA_CMD:
        {
            ret = Pack_MsgStreamMediaCmdResp(body_pos, buf_len-sizeof(MsgHeader), (StreamMediaCmdResp*)resp);
        }
        break;
    case MSG_ID_MEDIA_CLOSE:
        {
            ret = Pack_MsgStreamMediaCloseResp(body_pos, buf_len-sizeof(MsgHeader), (StreamMediaCloseResp*)resp);
        }
        break;
    case MSG_ID_DEV_MEDIA_OPEN:
        {
            ret = Pack_MsgDeviceMediaOpenResp(body_pos, buf_len-sizeof(MsgHeader), (DeviceMediaOpenResp*)resp);
        }
        break;
    case MSG_ID_DEV_MEDIA_CLOSE:
        {
            ret = Pack_MsgDeviceMediaCloseResp(body_pos, buf_len-sizeof(MsgHeader), (DeviceMediaCloseResp*)resp);
        }
        break;
    case MSG_ID_DEV_SNAP:
        {
            ret = Pack_MsgDeviceSnapResp(body_pos, buf_len-sizeof(MsgHeader), (DeviceSnapResp*)resp);
        }
        break;
    case MSG_ID_DEV_CTRL:
        {
            ret = Pack_MsgDeviceCtrlResp(body_pos, buf_len-sizeof(MsgHeader), (DeviceCtrlResp*)resp);
        }
        break;
    default:
        {
            ret = -3;
        }
        break;
    }

    if ( ret<0 )
    {
        return ret;
    }

    msg_size = ret + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = msg_id;
        header.msg_type = MSG_TYPE_RESP;
        header.msg_seq = msg_seq;
        (void)Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header);
    }

    return msg_size;
}