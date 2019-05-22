#ifndef __DNL_TRANSPORT_H__
#define  __DNL_TRANSPORT_H__

#include "comm_includes.h"

#include "dnl_def.h"
#undef  EXT
#ifndef __DNL_TRANSPORT_C__
#define EXT extern
#else
#define EXT
#endif

#define TP_TX_DEF_BUFF_LEN (8*1024+256)
#define TP_RX_DEF_BUFF_LEN (8*1024+256)

typedef enum tp_tcp_flag_e
{
	en_tp_flag_none,
	en_tp_flag_writing,
	en_tp_flag_reading,
	en_tp_flag_complete,
	en_tp_flag_err
}tp_tcp_flag_e;

typedef enum tp_tcp_state_e
{
	en_tcp_state_none,
	en_tcp_state_connecting,
	en_tcp_state_connected,
	en_tcp_state_disconnect,
	en_tcp_state_close
}tp_tcp_state_e;

typedef enum tp_rx_state_e
{
    en_tp_rx_recving = 1,
    en_tp_rx_complete,
    en_tp_rx_err
}tp_rx_state_e;

typedef struct dnl_sin_addr_info
{
	vos_uint32_t sin_addr;
	vos_uint16_t sin_port;
}dnl_sin_addr_info;

typedef struct dnl_transport_cfg_t
{
	vos_uint32_t        sock_type;
	vos_bool_t          is_block;
	dnl_sin_addr_info   peer_host;
}dnl_transport_cfg_t;

typedef struct dnl_transport_com_t
{
	vos_uint8_t*    buf;
	vos_size_t      max_size;
	vos_size_t      data_size;
	vos_size_t      sent_size; //仅用于tx，已发送的数据
	vos_bool_t      flag;
	unsigned        err_cnt;
	vos_status_t    status;
}dnl_transport_com_t;

typedef struct dnl_transport_buf_t
{
	vos_uint8_t*    buf;
	vos_ssize_t		max_size;
	vos_ssize_t		data_size;
	vos_bool_t		flag;			//对于接受消息：0 没有数据或解析成功，1 数据待解析; 对于发送消息： 0 没有数据或发送成功，1 数据待发送
}dnl_transport_buf_t;

typedef struct dnl_transport_data_t
{
	vos_sock_t                  sock;
	vos_uint32_t                sock_type;
	dnl_transport_com_t         tx;
	dnl_transport_com_t         rx;
	dnl_sin_addr_info           peer_host;
	tp_tcp_state_e              tcp_state;
}dnl_transport_data_t;

/**
 * @comment: 打开transport
 * @param: cfg 配置参数
 * @param: tp 句柄
 * @return: 成功返回0， 失败返回值<0.
 */
EXT int dnl_transport_open(dnl_transport_cfg_t* cfg, dnl_transport_data_t* tp);

/**
 * @comment: 关闭transport
 * @param: tp 句柄
 * @return: 成功返回0， 失败返回值<0.
 */
EXT int dnl_transport_close(dnl_transport_data_t* tp);

/**
 * @comment: 检查tcp连接
 * @param: tp 句柄
 * @return: 成功返回0， 失败返回值<0.
 */
EXT int dnl_transport_tcp_connect_check(dnl_transport_data_t* tp);

EXT int dnl_transport_send(dnl_transport_data_t* tp);
EXT int dnl_transport_recv(dnl_transport_data_t* tp);

EXT void dnl_transport_tx_clear(dnl_transport_data_t* tp);
EXT void dnl_transport_rx_clear(dnl_transport_data_t* tp);
EXT void dnl_transport_rtx_clear(dnl_transport_data_t* tp);

EXT tp_tcp_state_e dnl_transport_tcp_state( dnl_transport_data_t* tp );

EXT vos_bool_t dnl_transport_tx_enable( dnl_transport_data_t* tp );
EXT vos_bool_t dnl_transport_tx_writing( dnl_transport_data_t* tp );

EXT vos_bool_t dnl_transport_rx_enable( dnl_transport_data_t* tp );
EXT vos_bool_t dnl_transport_rx_complete( dnl_transport_data_t* tp );

/**
 * 接受消息处理
 * @param: tp       句柄
 * @param: msg_id   接受指定的消息ID
 * @param: msg_type 接受指定的消息类型
 * @param: msg_seq  接受指定的消息序号
 * @return: 接受中en_tp_rx_recving，接受完成en_tp_rx_complete， 接受错误en_tp_rx_err
 */
EXT int dnl_tp_rx_data(dnl_transport_data_t* tp, vos_uint32_t msg_id, vos_uint32_t msg_type, vos_uint32_t msg_seq);

/**
 * 发送消息处理
 * @param: tp       句柄
 * @param: msg_id   接受指定的消息ID
 * @param: msg_type 接受指定的消息类型
 * @param: arg1     发送响应消息，arg1为msg_seq；发送流请求消息，arg1为media_session_t的指针
 * @param: arg2     发送响应消息，arg2为对应的数据结构体
 * @return: 成功返回0， 失败返回值<0.
 */
EXT int dnl_tp_tx_data(dnl_transport_data_t* tp, int msg_id, int msg_type, void* arg1, void* arg2);


#endif
