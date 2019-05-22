#ifndef __DNL_STREAM_H__
#define __DNL_STREAM_H__

#include <stdio.h>
#include "comm_includes.h"
#include "dnl_def.h"
#include "DeviceSDK.h"
#include "dnl_transport.h"

#undef  EXT
#ifndef __DNL_STREAM_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL
#define NOP NOP
//#define  DEBUG_DUMP 0

//#pragma pack(1)

#if (OS_UCOS_II == 1)
#define MEDIA_QUEUE_SIZE    25//10//25
#define FRAME_SNED_SIZE     (2*1024)//(8*1024)
#else
#define MEDIA_QUEUE_SIZE    200//100//10//25
#define FRAME_SNED_SIZE     (8*1024)//(8*1024)	
#endif

#define	STREAM_TX_BUF_LEN		8*1024
#define	STREAM_RX_BUF_LEN		(1024 + 128)
#define	STREAM_TX_FRM_BUF_LEN   8*1024

typedef enum stream_session_status_e
{
    en_stream_session_on_init,
    en_stream_session_on_connecting,
    en_stream_session_on_play,
    en_stream_session_on_pause,
    en_stream_session_on_close
}stream_session_status_e;

typedef struct media_desc_t
{
    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint16_t session_type;
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;
    vos_bool_t is_audio_open;
    vos_bool_t is_video_open;
    vos_uint32_t stream_svr_list_len;
    addr_info stream_svr_list[10];
}media_desc_t;

typedef struct media_queue_data_t
{
    vos_uint8_t  frm_type;
    vos_uint32_t frm_id;
    vos_uint32_t frm_av_id;
    vos_uint32_t frm_base_time;
    vos_uint32_t frm_ts;
    vos_uint32_t frm_size;
    vos_uint32_t frm_offset;
    vos_uint32_t frm_data_size;
	char frm_data[FRAME_SNED_SIZE];
}media_frame_data_t;

typedef struct media_queue_t
{
	vos_uint32_t frm_nums;
	vos_uint32_t rd_idx;
	vos_uint32_t rw_idx;
	media_frame_data_t que[MEDIA_QUEUE_SIZE];
}media_queue_t;

typedef struct media_frame_t
{
    media_queue_t media_que;

    vos_time_val last_upload_time;
    vos_uint32_t last_frm_ts;
    vos_time_val last_adjust_tv;
    vos_uint32_t last_frame_offset;
    vos_uint32_t last_frame_ts;
}media_frame_t;

typedef struct media_session_t
{
    VOS_DECL_LIST_MEMBER(struct media_session_t);

    vos_bool_t running;
    vos_uint16_t retry_cnt;

    char session_id[MAX_MEDIA_SESSION_ID_LEN+1];
    vos_uint16_t session_type;
    
    vos_uint16_t channel_id;
    vos_uint8_t stream_id;

    vos_bool_t is_audio_open;
    vos_bool_t is_video_open;
    vos_bool_t is_pause;

	dnl_transport_data_t tp;
    media_frame_t media_frame;

    vos_uint32_t start_time;
    vos_uint32_t report_cycle;
    vos_uint32_t last_report_time;
    vos_uint32_t load;
    
    stream_session_status_e status;
    vos_uint32_t cont_seq;
    vos_uint32_t sub_seq;
    vos_uint32_t sub_start_time;

    vos_uint32_t begin_time;
    vos_uint32_t end_time;

    vos_uint32_t cur_stream_svr_idx;
    vos_uint32_t stream_svr_list_len;
    addr_info stream_svr_list[10];

    token_t token;

    vos_uint32_t req_seq;

#ifdef DEBUG_DUMP
    FILE* pf_stream_info;
    FILE* pf_stream_data;
#endif
}media_session_t;

typedef struct
{
    vos_bool_t running;

    int session_num;
    int live_session_num;
    media_session_t live_session_list;

}stream_endpt_t;

//流模块节点(全局)
EXT stream_endpt_t g_DnlStreamEp;

//设备流模块初始化/结束
EXT int dnl_stream_init(void);
EXT int dnl_stream_final(void);

//设备流模块一秒定时处理
EXT int stream_1s_timer(void);

//设备流模块主程序入口
EXT vos_thread_ret_t stream_run_proc(vos_thread_arg_t arg);

//设备APP层的流上报接口
EXT int stream_media_event_report(media_session_t *session, Dev_Stream_Frame_t *stream);

//流会话打开/关闭
EXT int stream_media_session_open(media_desc_t desc);
EXT int stream_media_session_close(char* session_id);

//获取流会话
EXT media_session_t* stream_get_media_session(vos_uint16_t session_type, vos_int16_t ch_id, vos_uint8_t stream_id);
EXT media_session_t* stream_get_media_session_by_id(char* session_id);
EXT media_session_t* stream_get_media_session_by_tp(dnl_transport_data_t* tp);
EXT vos_bool_t stream_media_session_is_open(char* session_id);

//编辑生成设备的请求消息序列化
EXT int stream_build_msg_media_connect_req(media_session_t *session, void *msg_buf, vos_size_t buf_len);
EXT int stream_build_msg_media_disconnect_req(media_session_t *session, void *msg_buf, vos_size_t buf_len);
EXT int stream_build_msg_media_status_report_req(media_session_t *session, void *msg_buf, vos_size_t buf_len);
EXT int stream_build_msg_media_eos_notify(media_session_t *session, void *msg_buf, vos_size_t buf_len);

//接收到流消息处理
EXT int stream_on_recv_msg(dnl_transport_data_t *tp, MsgHeader *header, void *msg_buf, vos_size_t msg_len);

#ifdef __DNL_STREAM_C__
//设备流会话状态处理
static int stream_session_on_tp_open(void *arg);
static int stream_session_on_exchange(void *arg);
static int stream_session_on_media_connect(void *arg);
static int stream_session_on_normal_run(void *arg);
static int stream_session_on_close(void *arg);

//流服务器的响应消息处理
static int stream_on_msg_media_connect_resp(media_session_t *session, StreamMediaConnectResp* resp);
static int stream_on_msg_media_disconnect_resp(media_session_t *session, StreamMediaDisconnectResp* resp);
static int stream_on_msg_media_status_report_resp(media_session_t *session, StreamMediaStatusResp* resp);

//流服务器的请求消息处理
static int stream_on_msg_media_play_req(media_session_t *session, MsgHeader* header, StreamMediaPlayReq* req);
static int stream_on_msg_media_pause_req(media_session_t *session, MsgHeader* header, StreamMediaPauseReq* req);
static int stream_on_msg_media_cmd_req(media_session_t *session, MsgHeader* header, StreamMediaCmdReq* req);
static int stream_on_msg_media_close_req(media_session_t *session, MsgHeader* header, StreamMediaCloseReq* req);
#endif //__DNL_STREAM_C__

VOS_END_DECL

#endif	//__DNL_STREAM_H__
