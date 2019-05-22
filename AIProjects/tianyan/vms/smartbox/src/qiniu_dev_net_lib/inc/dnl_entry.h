#ifndef __DNL_ENTRY_H__
#define __DNL_ENTRY_H__

#include "comm_includes.h"
#include "dnl_def.h"
#include "dnl_transport.h"
#include "DeviceSDK.h"

#undef  EXT
#ifndef __DNL_ENTRY_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

#define	ENTRY_TX_BUF_LEN		(8*1024 + 256)
#define	ENTRY_RX_BUF_LEN		(8*1024 + 256)
#define ENTRY_PIC_SNED_SIZE     (8*1024)

typedef struct entry_recv_msg_t
{
    vos_bool_t has_recv_msg_flag;
    vos_uint32_t recv_msg_seq;
    vos_uint32_t recv_msg_time;

    MsgHeader header;
    union
    {
        DeviceMediaOpenReq media_open_req;
        DeviceMediaCloseReq media_close_req;
        DeviceSnapReq snap_req;
        DeviceCtrlReq ctrl_req;

        DeviceLoginResp login_resp;
        DeviceAbilityReportResp ability_report_resp;
        DeviceStatusReportResp status_report_resp;
        DeviceAlarmReportResp alarm_report_resp;
    }body;
}entry_recv_msg_t;

typedef struct entry_media_status_report_t
{
    vos_uint32_t cur_index;
    vos_uint32_t max_list_size;
    DeviceMediaSessionStatus* status_list;
}entry_media_status_report_t;

typedef struct entry_alarm_t
{
    VOS_DECL_LIST_MEMBER(struct entry_alarm_t);
    vos_uint32_t type;
    vos_uint16_t channel_index;
    vos_uint8_t status;
}entry_alarm_t;

typedef struct entry_snap_i_t
{
    vos_bool_t snaping_flag;
    vos_int32_t pic_fmt;
    vos_uint32_t pic_size;
    vos_uint32_t pic_sent_size;
    vos_uint32_t pic_buffer_size;
    vos_uint8_t* pic_buffer;
}entry_snap_i_t;

typedef struct entry_snap_task_t
{
    vos_uint16_t max_channel_num;
    entry_snap_i_t* snap_task_tbl;
}entry_snap_task_t;

typedef struct entry_session_t
{
    vos_mutex_t* mutex;

    addr_info svr_addr;
    token_t token;

    //mode status info
    vos_bool_t in_main_proc;
    vos_int32_t cont_seq;
    vos_int32_t time_wait;
    vos_int32_t retry_cnt;

    vos_uint32_t start_time;
    vos_uint32_t report_cycle;
    vos_uint32_t last_report_time;

    dnl_transport_data_t tp;
    vos_uint32_t sub_seq;
    vos_uint32_t sub_start_time;

    //request sequence
    vos_uint32_t req_seq;

    //recv msg info
    entry_recv_msg_t recv_msg;

    //channel status change flag
    vos_bool_t channel_status_report;

    //media status report info
    entry_media_status_report_t media_status_report;

    //snap task info
    entry_snap_task_t snap_task;

    //alarm info
    entry_alarm_t alarm_list;

}entry_session_t;

typedef struct entry_endpt_t
{
    vos_bool_t running;

    //mode status info
    vos_int32_t mode_cont_seq;
    vos_int32_t mode_time_wait;
    vos_int32_t mode_retry_cnt;

    //session svr info
    entry_session_t session;
}entry_endpt_t;

EXT entry_endpt_t g_DnlEntryEp;
EXT Entry_Serv_t g_DnlEntryAddr;

EXT int dnl_entry_init();
EXT int dnl_entry_final();
EXT vos_thread_ret_t entry_run_proc(vos_thread_arg_t arg);
EXT void entry_1s_timeout(void);
EXT void entry_app_cb_set(Dev_Cmd_Cb_Func cb, void* user_data);

EXT int entry_build_msg_device_login_req(void *msg_buf, vos_size_t buf_len);
EXT int entry_build_msg_ability_report_req(void *msg_buf, vos_size_t buf_len);
EXT int entry_build_msg_status_report_req(void *msg_buf, vos_size_t buf_len);
EXT int entry_build_msg_alarm_report_req(void *msg_buf, vos_size_t buf_len);

//接收到会话消息处理
//EXT int entry_on_recv_msg(dnl_transport_data_t *tp, MsgHeader *header, void *msg_buf, vos_size_t msg_len);
EXT int entry_unpack_recv_msg( dnl_transport_data_t *tp );
EXT int entry_on_msg_device_login_resp(DeviceLoginResp* resp);

//媒体状态变化通知
EXT void entry_media_staus_chg(DeviceMediaSessionStatus status);
EXT void entry_media_session_close_notify(vos_uint16_t session_type, vos_uint16_t channel_id, vos_uint8_t stream_id);

#ifdef __DNL_ENTRY_C__
static int entry_mode_on_init(void *arg);
static int entry_mode_on_get_devid(void *arg);
static int entry_mode_on_get_access_token(void *arg);
static int entry_mode_on_get_session_svr(void *arg);
static int entry_mode_on_session(void *arg);
static int entry_mode_on_error(void *arg);

static int entry_session_on_tp_open(void *arg);
static int entry_session_on_exchange(void *arg);
static int entry_session_on_login(void *arg);
static int entry_session_on_ability_report(void *arg);
static int entry_session_on_normal_run(void* arg);
static int entry_session_on_error(void* arg);

static int entry_on_recv_msg();

static int entry_on_msg_media_open_req(entry_session_t *session, MsgHeader* header, DeviceMediaOpenReq* req);
static int entry_on_msg_media_close_req(entry_session_t *session, MsgHeader* header, DeviceMediaCloseReq* req);
static int entry_on_msg_snap_req(entry_session_t *session, MsgHeader* header, DeviceSnapReq* req);
static int entry_on_msg_device_ctrl_req(entry_session_t *session, MsgHeader* header, DeviceCtrlReq* req);

//static int entry_on_msg_device_login_resp(entry_session_t *session, DeviceLoginResp* resp);
static int entry_on_msg_device_ability_report_resp(entry_session_t *session, DeviceAbilityReportResp* resp);
static int entry_on_msg_device_status_report_resp(entry_session_t *session, DeviceStatusReportResp* resp);
static int entry_on_msg_device_alarm_report_resp(entry_session_t *session, DeviceAlarmReportResp* resp);

#endif	//__DNL_ENTRY_C__

VOS_END_DECL

#endif //__DNL_ENTRY_H__
