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
	vos_size_t      sent_size; //������tx���ѷ��͵�����
	vos_bool_t      flag;
	unsigned        err_cnt;
	vos_status_t    status;
}dnl_transport_com_t;

typedef struct dnl_transport_buf_t
{
	vos_uint8_t*    buf;
	vos_ssize_t		max_size;
	vos_ssize_t		data_size;
	vos_bool_t		flag;			//���ڽ�����Ϣ��0 û�����ݻ�����ɹ���1 ���ݴ�����; ���ڷ�����Ϣ�� 0 û�����ݻ��ͳɹ���1 ���ݴ�����
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
 * @comment: ��transport
 * @param: cfg ���ò���
 * @param: tp ���
 * @return: �ɹ�����0�� ʧ�ܷ���ֵ<0.
 */
EXT int dnl_transport_open(dnl_transport_cfg_t* cfg, dnl_transport_data_t* tp);

/**
 * @comment: �ر�transport
 * @param: tp ���
 * @return: �ɹ�����0�� ʧ�ܷ���ֵ<0.
 */
EXT int dnl_transport_close(dnl_transport_data_t* tp);

/**
 * @comment: ���tcp����
 * @param: tp ���
 * @return: �ɹ�����0�� ʧ�ܷ���ֵ<0.
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
 * ������Ϣ����
 * @param: tp       ���
 * @param: msg_id   ����ָ������ϢID
 * @param: msg_type ����ָ������Ϣ����
 * @param: msg_seq  ����ָ������Ϣ���
 * @return: ������en_tp_rx_recving���������en_tp_rx_complete�� ���ܴ���en_tp_rx_err
 */
EXT int dnl_tp_rx_data(dnl_transport_data_t* tp, vos_uint32_t msg_id, vos_uint32_t msg_type, vos_uint32_t msg_seq);

/**
 * ������Ϣ����
 * @param: tp       ���
 * @param: msg_id   ����ָ������ϢID
 * @param: msg_type ����ָ������Ϣ����
 * @param: arg1     ������Ӧ��Ϣ��arg1Ϊmsg_seq��������������Ϣ��arg1Ϊmedia_session_t��ָ��
 * @param: arg2     ������Ӧ��Ϣ��arg2Ϊ��Ӧ�����ݽṹ��
 * @return: �ɹ�����0�� ʧ�ܷ���ֵ<0.
 */
EXT int dnl_tp_tx_data(dnl_transport_data_t* tp, int msg_id, int msg_type, void* arg1, void* arg2);


#endif
