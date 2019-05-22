#define __DNL_ENTRY_C__

#include "vos_socket.h"
#include "cdiffie_hell_man.h"
#include "dnl_util.h"
#include "dnl_ctrl.h"
#include "dnl_dev.h"
#include "dnl_stream.h"
#include "dnl_log.h"
#include "dnl_timer.h"
#include "dnl_entry.h"

Dev_Cmd_Cb_Func g_DnlAppCb_func = NULL;
static void* g_DnlAppCb_userdata = NULL;

static const short cfgsvr_port = 80;

static const char* entry_pic_fmt_str[] = { "bmp", "jpg", "png" };

typedef enum entry_mode_cont_seq
{
    en_mode_seq_init,
    en_mode_seq_get_device_id,
    en_mode_seq_get_session_svr,
    en_mode_seq_session,
    en_mode_seq_error
}entry_mode_cont_seq;

static dnl_conttbl_t entry_mode_cont_tbl[] = 
{
    {&entry_mode_on_init,               en_mode_seq_get_device_id,      en_mode_seq_error,  0 }, //0
    {&entry_mode_on_get_devid,          en_mode_seq_get_session_svr,    en_mode_seq_error,  0 }, //1
    {&entry_mode_on_get_session_svr,    en_mode_seq_session,            en_mode_seq_error,  0 }, //2
    {&entry_mode_on_session,            en_mode_seq_session,		    en_mode_seq_error,  0 }, //3
    {&entry_mode_on_error,              en_mode_seq_init,		        en_mode_seq_error,  0 }  //4
};

typedef enum entry_session_cont_seq
{
    en_sess_seq_tp_open,
    en_sess_seq_exchange,
    en_sess_seq_login,
    en_sess_seq_ability_repoer,
    en_sess_seq_run_loop,
    en_sess_seq_error
}entry_session_cont_seq;

static dnl_conttbl_t entry_session_cont_tbl[] = 
{
    {&entry_session_on_tp_open,         en_sess_seq_exchange,		en_sess_seq_error,  0 }, //0
    {&entry_session_on_exchange,        en_sess_seq_login,		    en_sess_seq_error,  0 }, //1
    {&entry_session_on_login,           en_sess_seq_ability_repoer, en_sess_seq_error,  0 }, //2
    {&entry_session_on_ability_report,  en_sess_seq_run_loop,		en_sess_seq_error,  0 }, //3
    {&entry_session_on_normal_run,      en_sess_seq_run_loop,		en_sess_seq_error,  0 }, //4
    {&entry_session_on_error,           en_sess_seq_tp_open,		en_sess_seq_error,  0 }  //5
};

//会话通信打开处理
static int entry_session_transport_open(entry_session_t* session);

//会话状态上报
static int entry_session_status_report(entry_session_t *session);

//
static int get_device_id_byjson(const char* json, int json_len, char* device_id);
static int get_device_id_byweb(const Dev_OEM_Info_t *oem_info, char* device_id);

static int get_access_token_byjson(const char* json, int json_len, token_t* token);
static int get_access_token_byweb(const char *sn, token_t* token);

static int get_session_svr_byjson(const char* json, int json_len, addr_info* svr_addr, token_t* token);
static int get_session_svr_byweb(const char *sn, addr_info* svr_addr, token_t* token);

int dnl_entry_init()
{
	memset(&g_DnlEntryEp, 0x0, sizeof(g_DnlEntryEp));
	
    g_DnlEntryEp.mode_cont_seq = en_mode_seq_init;
    g_DnlEntryEp.running = TRUE;

    if( vos_mutex_create_recursive( NULL, &g_DnlEntryEp.session.mutex ) != VOS_SUCCESS )
    {
        return -1;
    }

    g_DnlEntryEp.session.cont_seq = en_sess_seq_tp_open;
    g_DnlEntryEp.session.recv_msg.has_recv_msg_flag = FALSE;

    g_DnlEntryEp.session.media_status_report.status_list = VOS_MALLOC_BLK_T(DeviceMediaSessionStatus, (g_DnlDevInfo.channel_num) );
    if( !g_DnlEntryEp.session.media_status_report.status_list )
    {
        return -2;
    }
    g_DnlEntryEp.session.media_status_report.max_list_size = g_DnlDevInfo.channel_num;
    g_DnlEntryEp.session.media_status_report.cur_index = 0;

    g_DnlEntryEp.session.snap_task.snap_task_tbl = VOS_MALLOC_BLK_T(entry_snap_i_t, g_DnlDevInfo.channel_num );
    if( !g_DnlEntryEp.session.snap_task.snap_task_tbl )
    {
        return -3;
    }
    memset(g_DnlEntryEp.session.snap_task.snap_task_tbl, 0x0, sizeof(entry_snap_i_t)*(g_DnlDevInfo.channel_num) ); //init
    g_DnlEntryEp.session.snap_task.max_channel_num = g_DnlDevInfo.channel_num;

    vos_list_init( &g_DnlEntryEp.session.alarm_list );

	return 0;
}

int dnl_entry_final()
{
	g_DnlEntryEp.running = FALSE;

    if( g_DnlEntryEp.session.media_status_report.status_list )
    {
        VOS_FREE_T(g_DnlEntryEp.session.media_status_report.status_list);
        g_DnlEntryEp.session.media_status_report.status_list = NULL;
        g_DnlEntryEp.session.media_status_report.max_list_size = 0;
        g_DnlEntryEp.session.media_status_report.cur_index = 0;
    }

    if( g_DnlEntryEp.session.snap_task.snap_task_tbl )
    {
        int i = 0;
        for( ; i<g_DnlEntryEp.session.snap_task.max_channel_num; ++i )
        {
            entry_snap_i_t* snap_task_i = &g_DnlEntryEp.session.snap_task.snap_task_tbl[i];
            if( !snap_task_i->pic_buffer )
            {
                VOS_FREE_T(snap_task_i->pic_buffer);
                snap_task_i->pic_buffer = NULL;
                snap_task_i->pic_buffer_size = 0;
            }
        }
        VOS_FREE_T(g_DnlEntryEp.session.snap_task.snap_task_tbl);
        g_DnlEntryEp.session.snap_task.snap_task_tbl = NULL;
        g_DnlEntryEp.session.snap_task.max_channel_num = 0;
    }

	return 0;
}

vos_thread_ret_t entry_run_proc(vos_thread_arg_t arg)
{
    int rst = RS_KP;
    
    do 
    {
        vos_uint32_t s_tick, e_tick, delt_tick;
        //vos_time_val tv = {0, 0};
        dnl_conttbl_t *mode_cont_tbl;
        vos_int32_t cur_mode_seq = 0;

        s_tick = vos_get_system_tick();
        cur_mode_seq = g_DnlEntryEp.mode_cont_seq;

        if ( !g_DnlEntryEp.running )
        {
            rst = RS_NG;
            break;
        }
        
        mode_cont_tbl = &entry_mode_cont_tbl[g_DnlEntryEp.mode_cont_seq];
        rst = mode_cont_tbl->func( &g_DnlEntryEp );
        if( rst == RS_OK )
        {
            g_DnlEntryEp.mode_cont_seq = mode_cont_tbl->ok_seq;
        }
        else if( rst == RS_NG )
        {
            g_DnlEntryEp.mode_cont_seq = mode_cont_tbl->ng_seq;
        }
        else if( rst == RS_KP )
        {
        }
        e_tick = vos_get_system_tick();
        delt_tick = e_tick - s_tick;
        if( delt_tick > 50 )
        {
            DNL_WARN_LOG(
                "entry session handle time too long, (%d, %d, %d), (%u, %u, %u)!\n",
                cur_mode_seq, g_DnlEntryEp.mode_retry_cnt, rst,
                s_tick, e_tick, delt_tick);
        }
    } while (0);

    return rst;
}

void entry_1s_timeout(void)
{
    if(g_DnlEntryEp.mode_time_wait > 0)
    {
        g_DnlEntryEp.mode_time_wait --;
    }

    if(g_DnlEntryEp.session.time_wait > 0)
    {
        g_DnlEntryEp.session.time_wait--;
    }

    //DNL_INFO_LOG("Invoke entry_1s_timeout()!");
}

void entry_app_cb_set(Dev_Cmd_Cb_Func cb, void* user_data)
{
    g_DnlAppCb_func = cb;
    g_DnlAppCb_userdata = user_data;
}

static int entry_mode_on_init(void *arg)
{
    int rst = RS_KP;
    do 
    {
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;
        if(!pEntryEp)
        {
            rst = RS_NG;
            break;
        }

        pEntryEp->mode_time_wait = 0;
        pEntryEp->mode_retry_cnt = 0;

        pEntryEp->session.cont_seq = en_sess_seq_tp_open;
        pEntryEp->session.sub_seq = 0;
        pEntryEp->session.sub_start_time = 0;

        rst = RS_OK;

    } while (0);
    
    return rst;
}
static int entry_mode_on_get_devid(void *arg)
{
    int rst = RS_KP;

    if( strlen(g_DnlDevInfo.dev_id) > 0 )
    {
        return RS_OK;
    }

    do 
    {
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;
        if(!pEntryEp)
        {
            rst = RS_NG;
            break;
        }

        if (pEntryEp->mode_time_wait)
        { 
            break;
        }

        if (get_device_id_byweb( &g_DnlDevInfo.oem_info, g_DnlDevInfo.dev_id ) < 0)
        {
            pEntryEp->mode_time_wait = 60;
            pEntryEp->mode_retry_cnt++;
            break;
        }
        else
        {
            pEntryEp->mode_retry_cnt = 0;
        }

        rst = RS_OK;

    } while (0);

    return rst;
}
#if 0
static int entry_mode_on_get_access_token(void *arg)
{
    int rst = RS_KP;
    do 
    {
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;
        if(!pEntryEp)
        {
            rst = RS_NG;
            break;
        }

        if (pEntryEp->mode_time_wait)
        {
            break;
        }

        if (get_access_token_byweb(g_DnlDevInfo.dev_id, &g_DnlEntryEp.session.token) < 0)
        {
            pEntryEp->mode_time_wait = 60;
            pEntryEp->mode_retry_cnt++;
            break;
        }
        else
        {
            pEntryEp->mode_retry_cnt = 0;
        }

        rst = RS_OK;

    } while (0);

    return rst;
}
#endif

static int entry_mode_on_get_session_svr(void *arg)
{
    int rst = RS_KP;
    do 
    {
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;
        if(!pEntryEp)
        {
            rst = RS_NG;
            break;
        }

        if (pEntryEp->mode_time_wait)
        {
            break;
        }

        if (get_session_svr_byweb(g_DnlDevInfo.dev_id, &g_DnlEntryEp.session.svr_addr, &g_DnlEntryEp.session.token) < 0)
        {
            pEntryEp->mode_time_wait = 60;
            pEntryEp->mode_retry_cnt++;
            rst = RS_NG;
            break;
        }
        else
        {
            pEntryEp->mode_retry_cnt = 0;
        }

        rst = RS_OK;

    } while (0);

    return rst;
}

static int entry_mode_on_session(void *arg)
{
    int rst = RS_KP;
    do 
    {
        dnl_conttbl_t *session_cont_tbl;
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;

        if(!pEntryEp)
        {
            rst = RS_NG;
            break;
        }

        session_cont_tbl = &entry_session_cont_tbl[g_DnlEntryEp.session.cont_seq];
        rst = session_cont_tbl->func( &g_DnlEntryEp.session );
        if( rst == RS_OK )
        {
            g_DnlEntryEp.session.cont_seq = session_cont_tbl->ok_seq;
        }
        else if( rst == RS_NG )
        {
            if(g_DnlEntryEp.session.cont_seq == 5) //entry_session_on_error
            {
                pEntryEp->mode_time_wait = 30;
                break;//
            }
            else
            {
                rst = RS_KP;
                g_DnlEntryEp.session.cont_seq = session_cont_tbl->ng_seq;
            }
        }
        else if( rst == RS_KP )
        {
        }

    } while (0);
    return rst;
}
static int entry_mode_on_error(void *arg)
{
    int rst = RS_OK;
    do 
    {
        entry_endpt_t* pEntryEp = (entry_endpt_t*)arg;

        if(!pEntryEp)
        {
            break;
        }
        if( pEntryEp->mode_time_wait > 0 )
        {
            rst = RS_KP;
            break;
        }

        pEntryEp->mode_cont_seq = en_mode_seq_init;
        //pEntryEp->mode_time_wait = 0;
        pEntryEp->mode_retry_cnt = 0;

    } while (0);

    return rst;
}

static int get_device_id_byjson(const char* json, int json_len, char* device_id)
{
    cJSON *pjson = NULL;
    cJSON *js_tmp = NULL;
    cJSON *js_devid = NULL;

    do 
    {
        if (!json || !device_id)
        {
            break;
        }

        pjson = cJSON_Parse(json);
        if (!pjson)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "code" );
        if (!js_tmp)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "message" );
        if (!js_tmp)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "data" );
        if (!js_tmp)
        {
            break;
        }

        js_devid = cJSON_GetObjectItem( js_tmp, "device_id" );
        if (!js_devid || !js_devid->valuestring)
        {
            break;
        }

        strcpy(device_id, js_devid->valuestring);
        cJSON_Delete( pjson );
        return 0;
    } while (false);
    return -1;
}

static int get_device_id_byweb(const Dev_OEM_Info_t *oem_info, char* device_id)
{
    int ret = -1;

    char* http_url = NULL;
    char* clear_text = NULL;
    int des_len = 512;
    char* des_text = NULL;
    char* encode_text = NULL;

    ghttp_request* pRequest = NULL;
    char* json_context = NULL;
    int json_len = 0;
    int context_len = 0;
    char* pWebContent = NULL;

    do 
    {
        if (!oem_info || !device_id)
        {
            break;
        }

        DNL_DEBUG_LOG("get_device_id_byweb, OEM_name=%s, MAC=%s", oem_info->OEM_name, oem_info->MAC);

        http_url = VOS_MALLOC_BLK_T(char, 1024);
        if(!http_url)
        {
            break;
        }

        memset( http_url, 0x0, 1024 );

        clear_text = VOS_MALLOC_BLK_T(char, 256);
        if(!clear_text)
        {
            break;
        }

        memset( clear_text, 0x0, 256 );

#if 0
        snprintf(clear_text, 256, "cmd=get_device_sn&oem_head=%s&oem_sn=%s&timestamp=%u", sn_info->OEM_name, sn_info->SN, time(NULL));

        des_text = VOS_MALLOC_BLK_T(char, des_len);
        if(!des_text)
        {
            break;
        }

        memset( des_text, 0x0, des_len );
        ret = DES_Encrypt( (unsigned char*)clear_text, strlen(clear_text), (unsigned char*)ANYAN_DEFAULT_KEY, 8, (unsigned char*)des_text, des_len, &des_len );
        if (ret < 0)
        {
            break;
        }

        encode_text = VOS_MALLOC_BLK_T(char, 1024);
        if(!encode_text)
        {
            break;
        }

        memset( encode_text, 0x0, 1024 );

        ret = hex_encode((unsigned char*)des_text, des_len, encode_text);
        if ( ret < 0 )
        {
            DNL_ERROR_LOG("hex_encode failed, %d\n", des_len);
            break;
        }

        snprintf(http_url, 1024, "%s?params=%s",url, encode_text);
#else
        snprintf( http_url, 1024, "http://%s:%d/gen_device_id?oem_id=%s&sn=%s",
            g_DnlEntryAddr.ip, g_DnlEntryAddr.port, oem_info->OEM_name, oem_info->SN );
#endif
        ret = -1;

        pRequest = request_webserver_new();
        DNL_DEBUG_LOG("maloc request object, %p", pRequest);
        if (!pRequest)
        {
            break;
        }

        {
            pWebContent = request_webserver_content(http_url, pRequest, &context_len);
            if (!pWebContent)
            {
                break;
            }
            json_context = VOS_MALLOC_BLK_T(char, context_len+1);
            if (!json_context)
            {
                break;
            }
            memset(json_context, 0x0, context_len+1);

            {
                json_len = json_info_decode(pWebContent, context_len, json_context, context_len);
                if (json_len <= 0)
                {
                    break;
                }

                if (get_device_id_byjson(json_context, json_len, device_id) < 0)
                {
                    break;
                }

                DNL_DEBUG_LOG("get_device_id_byweb, device_id=%s", device_id);
            }        

        }
        
        ret = 0;
    } while (false);

    if (pRequest)
    {
        DNL_DEBUG_LOG("free request object, %p", pRequest);
        request_webserver_destroy(pRequest);
    }

    if (json_context)
    {
        VOS_FREE_T(json_context);
    }

    if (encode_text)
    {
        VOS_FREE_T(encode_text);
    }

    if (des_text)
    {
        VOS_FREE_T(des_text);
    }

    if (clear_text)
    {
        VOS_FREE_T(clear_text);
    }

    if (http_url)
    {
        VOS_FREE_T(http_url);
    }

    return ret;
}

static int get_access_token_byjson(const char* json, int json_len, token_t* token)
{
    cJSON *pjson = NULL;
    cJSON *js_tmp = NULL;
    cJSON *js_token = NULL;
    do 
    {
        if (!json || !token)
        {
            break;
        }

        pjson = cJSON_Parse(json);
        if (!pjson)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "code" );
        if (!js_tmp)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "message" );
        if (!js_tmp)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "data" );
        if (!js_tmp)
        {
            break;
        }

        js_token = cJSON_GetObjectItem( js_tmp, "access_token" );
        if (!js_token || !js_token->valuestring)
        {
            break;
        }

        {
            int token_len = sizeof(token->token_bin);
            strncpy((char*)token->token_bin, js_token->valuestring, token_len);
            token->token_bin_length = strlen((char*)token->token_bin);
        }
        cJSON_Delete( pjson );
        return 0;
    } while (false);
    return -1;
}

static int get_access_token_byweb(const char *device_id, token_t* token)
{
    int ret = -1;

    char* http_url = NULL;
    char* clear_text = NULL;
    int des_len = 512;
    char* des_text = NULL;
    char* encode_text = NULL;

    ghttp_request* pRequest = NULL;
    char* json_context = NULL;

    do 
    {
        if (!device_id || !token)
        {
            break;
        }

        DNL_DEBUG_LOG("get_access_token_byweb, device_id=%s\n", device_id);

        http_url = VOS_MALLOC_BLK_T(char, 1024);
        if(!http_url)
        {
            break;
        }

        memset( http_url, 0x0, 1024 );

        clear_text = VOS_MALLOC_BLK_T(char, 256);
        if(!clear_text)
        {
            break;
        }

        memset( clear_text, 0x0, 256 );
#if 0
        snprintf(clear_text, 256, "cmd=get_access_token&device_sn=%s&timestamp=%u", sn, time(NULL));

        des_text = VOS_MALLOC_BLK_T(char, des_len);
        if(!des_text)
        {
            break;
        }

        ret = DES_Encrypt( (unsigned char*)clear_text, strlen(clear_text), (unsigned char*)ANYAN_DEFAULT_KEY, 8, (unsigned char*)des_text, des_len, &des_len );

        if (ret < 0)
        {
            break;
        }

        encode_text = VOS_MALLOC_BLK_T(char, 1024);
        if(!encode_text)
        {
            break;
        }

        ret = hex_encode((unsigned char*)des_text, des_len, encode_text);
        if ( ret < 0 )
        {
            DNL_ERROR_LOG("hex_encode failed, %d\n", des_len);
            break;
        }

        snprintf(http_url, 1024, "%s?params=%s",url, encode_text);
#else
        snprintf( http_url, 1024, "http://%s:%d/get_access_token?device_id=%s",
            g_DnlEntryAddr.ip, g_DnlEntryAddr.port, device_id );
#endif
        ret = -1;

        pRequest = request_webserver_new();
        DNL_DEBUG_LOG("alloc request object, %p", pRequest);
        if (!pRequest)
        {
            break;
        }

        {
            int context_len = 0;
            char* pWebContent = request_webserver_content(http_url, pRequest, &context_len);
            if (!pWebContent)
            {
                break;
            }

            json_context = VOS_MALLOC_BLK_T(char, context_len+1);
            if (!json_context)
            {
                break;
            }
            memset(json_context, 0x0, context_len+1);

            {
                int json_len = json_info_decode(pWebContent, context_len, json_context, context_len);
                if (json_len <= 0)
                {
                    break;
                }

                if (get_access_token_byjson(json_context, json_len, token) < 0)
                {
                    break;
                }
            }
            DNL_DEBUG_LOG("get_access_token_byweb, token=%s\n", token->token_bin);
        }
        
        ret = 0;
    } while (false);

    if (pRequest)
    {
        DNL_DEBUG_LOG("free request object, %p", pRequest);
        request_webserver_destroy(pRequest);
    }

    if (json_context)
    {
        VOS_FREE_T(json_context);
    }

    if (encode_text)
    {
        VOS_FREE_T(encode_text);
    }

    if (des_text)
    {
        VOS_FREE_T(des_text);
    }

    if (clear_text)
    {
        VOS_FREE_T(clear_text);
    }

    if (http_url)
    {
        VOS_FREE_T(http_url);
    }

    return ret;
}

static int get_session_svr_byjson(const char* json, int json_len, addr_info* svr_addr, token_t* token)
{
    cJSON *pjson = NULL;
    cJSON *js_tmp = NULL;
    cJSON *js_item = NULL;
    cJSON *js_ip = NULL;
    cJSON *js_port = NULL;
    cJSON *js_token = NULL;

    do 
    {
        if (!json || !svr_addr)
        {
            break;
        }

        pjson = cJSON_Parse(json);
        if (!pjson)
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "code" );
        if (!js_tmp || (js_tmp->valueint != 0) )
        {
            break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "msg" );
        if (!js_tmp)
        {
            //break;
        }

        js_tmp = cJSON_GetObjectItem( pjson, "data" );
        if (!js_tmp)
        {
            break;
        }

        js_ip = cJSON_GetObjectItem( js_tmp, "ip" );
        if (!js_ip || !js_ip->valuestring || !strlen(js_ip->valuestring) )
        {
            break;
        }

        js_port = cJSON_GetObjectItem( js_tmp, "port" );
        if ( !js_port || !js_port->valueint )
        {
            break;
        }

        {
            vos_in_addr in_addr;
            strcpy(svr_addr->IP, js_ip->valuestring);
            in_addr = vos_inet_addr2(svr_addr->IP);
            svr_addr->sin_addr = in_addr.s_addr;
            svr_addr->port = js_port->valueint;
        }

        js_token = cJSON_GetObjectItem( js_tmp, "token" );
        if (js_token && js_token->valuestring)
        {
            int token_len = sizeof(token->token_bin);
            strncpy((char*)token->token_bin, js_token->valuestring, token_len);
            token->token_bin_length = strlen((char*)token->token_bin);
        }
        
        cJSON_Delete( pjson );
        return 0;
    } while (false);
    return -1;
}

static int get_session_svr_byweb(const char *device_id, addr_info* svr_addr, token_t* token)
{
    int ret = -1;

    char* http_url = NULL;
    char* clear_text = NULL;
    int des_len = 512;
    char* des_text = NULL;
    char* encode_text = NULL;

    ghttp_request* pRequest = NULL;
    char* json_context = NULL;

    do
    {
        if (!device_id || !svr_addr)
        {
            break;
        }

        DNL_DEBUG_LOG("get_session_svr_byweb, device_id=%s\n", device_id);

        http_url = VOS_MALLOC_BLK_T(char, 1024);
        if(!http_url)
        {
            break;
        }

        memset( http_url, 0x0, 1024 );

        clear_text = VOS_MALLOC_BLK_T(char, 256);
        if(!clear_text)
        {
            break;
        }

        memset( clear_text, 0x0, 256 );
#if 0
        snprintf(clear_text, 256, "cmd=get_session_server&device_sn=%s&timestamp=%u", sn, time(NULL));

        des_text = VOS_MALLOC_BLK_T(char, des_len);
        if(!des_text)
        {
            break;
        }

        ret = DES_Encrypt( (unsigned char*)clear_text, strlen(clear_text), (unsigned char*)ANYAN_DEFAULT_KEY, 8, (unsigned char*)des_text, des_len, &des_len );

        if (ret < 0)
        {
            break;
        }

        encode_text = VOS_MALLOC_BLK_T(char, 1024);
        if(!encode_text)
        {
            break;
        }

        ret = hex_encode((unsigned char*)des_text, des_len, encode_text);
        if ( ret < 0 )
        {
            DNL_ERROR_LOG("hex_encode failed, %d\n", des_len);
            break;
        }

        snprintf(http_url, 1024, "%s?params=%s",url, encode_text);
#else
        snprintf(http_url, 1024, "http://%s:%d/get_session_server?device_id=%s",
            g_DnlEntryAddr.ip, g_DnlEntryAddr.port, device_id);
#endif
        ret = -1;

        pRequest = request_webserver_new();
        if (!pRequest)
        {
            break;
        }

        {
            int context_len = 0;
            char* pWebContent = request_webserver_content(http_url, pRequest, &context_len);
            if (!pWebContent)
            {
                break;
            }

            json_context = VOS_MALLOC_BLK_T(char, context_len+1);
            if (!json_context)
            {
                break;
            }
            memset(json_context, 0x0, context_len+1);

            {
                int json_len = json_info_decode(pWebContent, context_len, json_context, context_len);
                if (json_len <= 0)
                {
                    break;
                }

                if (get_session_svr_byjson(json_context, json_len, svr_addr, token) < 0)
                {
                    break;
                }
            }

            DNL_DEBUG_LOG("get_session_svr_byweb, ip=%s, port=%u\n", svr_addr->IP, svr_addr->port);
        }
        ret = 0;
    } while (false);

    if (pRequest)
    {
        request_webserver_destroy(pRequest);
    }

    if (json_context)
    {
        VOS_FREE_T(json_context);
    }

    if (encode_text)
    {
        VOS_FREE_T(encode_text);
    }

    if (des_text)
    {
        VOS_FREE_T(des_text);
    }

    if (clear_text)
    {
        VOS_FREE_T(clear_text);
    }

    if (http_url)
    {
        VOS_FREE_T(http_url);
    }

    return ret;
}

static int entry_session_transport_open(entry_session_t* session)
{
    dnl_transport_cfg_t tp_cfg;

    if(!session)
    {
        return -1;
    }

    memset(&tp_cfg, 0x0, sizeof(tp_cfg));
    {
        addr_info* svr_addr = &session->svr_addr;
        tp_cfg.peer_host.sin_addr = svr_addr->sin_addr;
        tp_cfg.peer_host.sin_port = vos_htons( svr_addr->port );
        tp_cfg.sock_type = SOCK_STREAM;
    }
    
    if( dnl_transport_open(&tp_cfg, &session->tp) != 0)
    {
        return -1;
    }

    if ( session->tp.tx.buf == NULL )
    {
        session->tp.tx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, ENTRY_TX_BUF_LEN);
        session->tp.tx.max_size = ENTRY_TX_BUF_LEN;
    }

    if ( session->tp.rx.buf == NULL )
    {
        session->tp.rx.buf = VOS_MALLOC_BLK_T(vos_uint8_t, ENTRY_RX_BUF_LEN);
        session->tp.rx.max_size = ENTRY_RX_BUF_LEN;
    }

    return 0;
}

void entry_media_staus_chg(DeviceMediaSessionStatus status)
{
    vos_uint32_t cur_index = 0;
    if ( !g_DnlEntryEp.session.in_main_proc )
    {
        return;
    }

    cur_index = g_DnlEntryEp.session.media_status_report.cur_index;
    if( cur_index >= g_DnlEntryEp.session.media_status_report.max_list_size )
    {
        vos_uint32_t new_list_size = g_DnlEntryEp.session.media_status_report.max_list_size + g_DnlDevInfo.channel_num/2;
        DeviceMediaSessionStatus* new_status_list = VOS_MALLOC_BLK_T(DeviceMediaSessionStatus, new_list_size);
        if(!new_status_list)
        {
            DNL_ERROR_LOG( "realloc media status list failed, curr_size=%u, new_size=%u\n", 
                cur_index, new_list_size );
            return;
        }
        else
        {
            memcpy( new_status_list, g_DnlEntryEp.session.media_status_report.status_list, sizeof(DeviceMediaSessionStatus)*cur_index );
            
            VOS_FREE_T(g_DnlEntryEp.session.media_status_report.status_list);
            g_DnlEntryEp.session.media_status_report.status_list = new_status_list;
        }
    }

    memcpy(&g_DnlEntryEp.session.media_status_report.status_list[cur_index], &status, sizeof(DeviceMediaSessionStatus));
    g_DnlEntryEp.session.media_status_report.cur_index ++;
}

void entry_media_session_close_notify(vos_uint16_t session_type, vos_uint16_t channel_id, vos_uint8_t stream_id)
{
    //notify app live open
    if( session_type == MEDIA_SESSION_TYPE_LIVE )
    {
        Dev_Cmd_Param_t cmd_param;
        cmd_param.cmd_type = EN_CMD_LIVE_CLOSE;
        cmd_param.channel_index = channel_id;
        cmd_param.cmd_args[0] = stream_id;
        g_DnlAppCb_func(g_DnlAppCb_userdata, &cmd_param);
    }
}

int entry_unpack_recv_msg( dnl_transport_data_t *tp )
{
    int ret = 0;
    char* body_pos = NULL;
    vos_size_t body_len = 0;
    entry_session_t *session = &g_DnlEntryEp.session;
    entry_recv_msg_t* recv_msg = &session->recv_msg;

    do 
    {
        MsgHeader* header = &recv_msg->header;

        if ( Unpack_MsgHeader(tp->rx.buf, sizeof(MsgHeader), &session->recv_msg.header) < 0 )
        {
            return -3;
        }

        body_pos = (char*)tp->rx.buf + sizeof(MsgHeader);
        body_len = tp->rx.data_size - sizeof(MsgHeader);

        if(header->msg_type == MSG_TYPE_REQ)
        {
            switch(header->msg_id)
            {
            case MSG_ID_DEV_MEDIA_OPEN:
                {
                    if ( Unpack_MsgDeviceMediaOpenReq( body_pos, body_len, &recv_msg->body.media_open_req) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            case MSG_ID_DEV_MEDIA_CLOSE:
                {
                    if ( Unpack_MsgDeviceMediaCloseReq( body_pos, body_len, &recv_msg->body.media_close_req) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            case MSG_ID_DEV_SNAP:
                {
                    if ( Unpack_MsgDeviceSnapReq( body_pos, body_len, &recv_msg->body.snap_req) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            case MSG_ID_DEV_CTRL:
                {
                    if ( Unpack_MsgDeviceCtrlReq( body_pos, body_len, &recv_msg->body.ctrl_req) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            default:
                {
                    ret = -5;
                }
                break;
            }
        }
        else if(session->recv_msg.header.msg_type == MSG_TYPE_RESP)
        {
            switch(session->recv_msg.header.msg_id)
            {
            /*case MSG_ID_DEV_LOGIN:
                {
                    if ( Unpack_MsgDeviceLoginResp( body_pos, body_len, &recv_msg->body.login_resp) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;*/
            case MSG_ID_DEV_ABILITY_REPORT:
                {
                    if ( Unpack_MsgDeviceAbilityReportResp( body_pos, body_len, &recv_msg->body.ability_report_resp) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            case MSG_ID_DEV_STATUS_REPORT:
                {
                    if ( Unpack_MsgDeviceStatusReportResp( body_pos, body_len, &recv_msg->body.status_report_resp) < 0 )
                    {
                        ret = -4;
                        break;
                    }
                }
                break;
            case MSG_ID_DEV_ALARM_REPORT:
                {

                }
                break;
            case MSG_ID_DEV_PIC_UPLOAD_REPORT:
                {

                }
                break;
            default:
                {
                    ret = -5;
                }
                break;
            }
        }
        else
        {
            ret = -6;
        }

    } while (0);

    if( ret == 0 )
    {
        recv_msg->has_recv_msg_flag = true;
        recv_msg->recv_msg_seq = 0;
        recv_msg->recv_msg_time = vos_get_system_tick_sec();
    }

    return ret;
}

static int entry_on_recv_msg()
{
    int ret = RS_KP;
    entry_session_t* session = &g_DnlEntryEp.session;
    entry_recv_msg_t* recv_msg = &session->recv_msg;

    do 
    {
        
        MsgHeader* header = &recv_msg->header;

        if( !recv_msg->has_recv_msg_flag )
        {
            ret = RS_OK;
            break;
        }

        if(header->msg_type == MSG_TYPE_REQ)
        {
            switch(header->msg_id)
            {
            case MSG_ID_DEV_MEDIA_OPEN:
                {
                    ret = entry_on_msg_media_open_req(session, header, &recv_msg->body.media_open_req);
                }
                break;
            case MSG_ID_DEV_MEDIA_CLOSE:
                {
                    ret = entry_on_msg_media_close_req(session, header, &recv_msg->body.media_close_req);
                }
                break;
            case MSG_ID_DEV_SNAP:
                {
                    ret = entry_on_msg_snap_req(session, header, &recv_msg->body.snap_req);
                }
                break;
            case MSG_ID_DEV_CTRL:
                {
                    ret = entry_on_msg_device_ctrl_req(session, header, &recv_msg->body.ctrl_req);
                }
                break;
            }
        }
        else if(header->msg_type == MSG_TYPE_RESP)
        {
            switch(header->msg_id)
            {
            /*case MSG_ID_DEV_LOGIN:
                {
                    ret = entry_on_msg_device_login_resp(session, &recv_msg->body.login_resp);
                }
                break;*/
            case MSG_ID_DEV_ABILITY_REPORT:
                {
                    ret = entry_on_msg_device_ability_report_resp(session, &recv_msg->body.ability_report_resp);
                }
                break;
            case MSG_ID_DEV_STATUS_REPORT:
                {
                    ret = entry_on_msg_device_status_report_resp(session, &recv_msg->body.status_report_resp);
                }
                break;
            case MSG_ID_DEV_ALARM_REPORT:
                {
                    ret = entry_on_msg_device_alarm_report_resp(session, &recv_msg->body.alarm_report_resp);;
                }
                break;
            case MSG_ID_DEV_PIC_UPLOAD_REPORT:
                {
                    ret = RS_OK;
                }
                break;
            }
        }
        else
        {
            ret = RS_NG;
        }

    } while (0);

    if( ret == RS_OK )
    {
        recv_msg->has_recv_msg_flag = false;
        recv_msg->recv_msg_seq = 0;
        recv_msg->recv_msg_time = 0;
    }

    return ret;
}

static int entry_session_on_tp_open(void *arg)
{
    int rst = RS_KP;
    vos_uint32_t cur_tim;
    entry_session_t* session = (entry_session_t*)arg;

    if( !session )
    {
        DNL_ERROR_LOG("entry_session_on_tp_open-->connector is nil!\n");
        return RS_NG;
    }

    switch(session->sub_seq)
    {
    case 0:
        session->start_time = vos_get_system_tick();
        if(entry_session_transport_open(session) < 0)
        {
            DNL_ERROR_LOG("entry_session_on_tp_open-->open failed!\n");
            rst = RS_NG;
        }
        else
        {
            if(session->tp.tcp_state == en_tcp_state_connected)
            {
                rst = RS_OK;
                DNL_INFO_LOG("entry_session_on_tp_open-->open success, sock_fd=%d.\n", session->tp.sock);
            }
            else if(session->tp.tcp_state == en_tcp_state_connecting)
            {
                session->sub_start_time = vos_get_system_tick_sec();
                session->sub_seq ++;
                rst = RS_KP;
                DNL_ERROR_LOG("entry_session_on_tp_open-->connecting, sock_fd=%d.\n", session->tp.sock);
            }
            else
            {
                rst = RS_NG;
                DNL_ERROR_LOG("entry_session_on_tp_open-->open failed, tcp_state=%d!\n", session->tp.tcp_state);
            }
        }
        break;
    case 1:
        if(dnl_transport_tcp_connect_check(&session->tp) == 0)
        {
            rst = RS_OK;
            DNL_ERROR_LOG("entry_session_on_tp_open-->open success, sock_fd=%d.\n", session->tp.sock);
        }
        else
        {
            cur_tim = vos_get_system_tick_sec();
            if( (cur_tim - session->sub_start_time) > 10 )
            {
                rst = RS_NG;
                DNL_ERROR_LOG("entry_session_on_tp_open-->open timeout, sock_fd=%d.\n", session->tp.sock);
            }
            else
            {
                rst = RS_KP;
            }
        }
        break;
    default:
        rst = RS_NG;
        DNL_ERROR_LOG("entry session open failed, subseq=%d!\n", session->sub_seq);
        break;
    }

    return rst;
}

static int entry_session_on_exchange(void *arg)
{
    int rst = RS_KP;
    entry_session_t* session = (entry_session_t*)arg;

    do 
    {
        vos_uint32_t cur_tim;
        tp_tcp_state_e tcp_state;

        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "stream_on_exchange-->01 error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );

            rst = RS_NG;
            break;
        }

        if( !STREAM_ENCRYPT ) //不加密无需密钥交换
        {
            rst = RS_OK;
            break;
        }

        switch(session->sub_seq)
        {
        case 0:
            {
                vos_uint32_t req_seq = ++session->req_seq;
                if(dnl_tp_tx_data(&session->tp, MSG_ID_EXCHANGE_KEY, MSG_TYPE_REQ, &req_seq, 0) == 0)
                {
                    session->sub_seq++;
                    session->sub_start_time = vos_get_system_tick_sec();
                    rst = RS_KP;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        case 1:
            {
                int ret = dnl_tp_rx_data(&session->tp, MSG_ID_EXCHANGE_KEY, MSG_TYPE_RESP, session->req_seq);
                if(ret == en_tp_rx_recving)
                {
                    cur_tim = vos_get_system_tick_sec();
                    if( (cur_tim - session->sub_start_time) > 10)
                    {
                        rst = RS_NG;
                        DNL_ERROR_LOG("wait exchange key response timeout, sock_fd=%d.\n", session->tp.sock);
                    }
                }
                else if(ret == en_tp_rx_complete)
                {
                    session->sub_seq = 0;
                    session->sub_start_time = 0;
                    rst = RS_OK;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        default:
            {
                session->sub_seq = 0;
                DNL_ERROR_LOG("stream_on_exchange-->default, sock_fd=%d, subseq=%d.\n", session->tp.sock, session->sub_seq);
            }            
            break;
        }

    } while (0);

    return rst;
}

static int entry_session_on_login(void *arg)
{
    int rst = RS_KP;
    vos_uint32_t cur_tim;
    tp_tcp_state_e tcp_state;

    entry_session_t* session = (entry_session_t*)arg;

    do 
    {
        if( !session )
        {
            rst = RS_NG;
            break;
        }

        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            rst = RS_NG;
            break;
        }

        switch(session->sub_seq)
        {
        case 0:
            {
                vos_uint32_t req_seq = ++session->req_seq;
                if(dnl_tp_tx_data(&session->tp, MSG_ID_DEV_LOGIN, MSG_TYPE_REQ, &req_seq, 0) == 0)
                {
                    session->sub_seq ++;
                    session->sub_start_time = vos_get_system_tick_sec();
                    rst = RS_KP;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        case 1:
            {
                int ret = dnl_tp_rx_data(&session->tp, MSG_ID_DEV_LOGIN, MSG_TYPE_RESP, session->req_seq);
                if(ret == en_tp_rx_recving)
                {
                    cur_tim = vos_get_system_tick_sec();
                    if( (cur_tim - session->sub_start_time) > 10)
                    {
                        rst = RS_NG;
                        DNL_ERROR_LOG("wait login response timeout, sock_fd=%d.\n", session->tp.sock);
                    }
                }
                else if(ret == en_tp_rx_complete)
                {
                    session->sub_seq = 0;
                    session->sub_start_time = 0;
                    rst = RS_OK;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        default:
            {
                session->sub_seq = 0;
                DNL_ERROR_LOG("stream_on_login-->default, sock_fd=%d, subseq=%d.\n", session->tp.sock, session->sub_seq);
            }
            break;
        }
    } while (false);

    return rst;
}

static int entry_session_on_ability_report(void *arg)
{
    int rst = RS_KP;
    vos_uint32_t cur_tim;
    tp_tcp_state_e tcp_state;

    entry_session_t* session = (entry_session_t*)arg;

    do 
    {
        if( !session )
        {
            rst = RS_NG;
            break;
        }

        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            rst = RS_NG;
            break;
        }

        switch(session->sub_seq)
        {
        case 0:
            {
                vos_uint32_t req_seq = ++session->req_seq;
                if(dnl_tp_tx_data(&session->tp, MSG_ID_DEV_ABILITY_REPORT, MSG_TYPE_REQ, &req_seq, 0) == 0)
                {
                    session->sub_seq ++;
                    session->sub_start_time = vos_get_system_tick_sec();
                    rst = RS_KP;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        case 1:
            {
                int ret = dnl_tp_rx_data(&session->tp, MSG_ID_DEV_ABILITY_REPORT, MSG_TYPE_RESP, session->req_seq);
                if(ret == en_tp_rx_recving)
                {
                    cur_tim = vos_get_system_tick_sec();
                    if( (cur_tim - session->sub_start_time) > 10)
                    {
                        rst = RS_NG;
                        DNL_ERROR_LOG("wait ability report response timeout, sock_fd=%d.\n", session->tp.sock);
                    }
                }
                else if(ret == en_tp_rx_complete)
                {
                    session->sub_seq = 0;
                    session->sub_start_time = 0;
                    session->in_main_proc = TRUE;
                    rst = RS_OK;
                }
                else
                {
                    rst = RS_NG;
                }
            }
            break;
        default:
            {
                session->sub_seq = 0;
                DNL_ERROR_LOG("stream_on_report-->default, sock_fd=%d, subseq=%d.\n", session->tp.sock, session->sub_seq);
            }            
            break;
        }

    } while (false);

    return rst;
}

static int entry_session_status_report(entry_session_t *session)
{
    vos_time_val cur_tim;
    vos_uint32_t repcycle = 10;
    vos_bool_t channel_report = false;
    int ret = 0;

    // for test reconnect
    //if(vos_get_system_tick() - session->start_time > 60*1000)
    //{
    //    return -1;
    //}

    if ( session->report_cycle == 0 || session->report_cycle > 30 )
    {
        repcycle = 30;
    }
    else
    {
        repcycle = session->report_cycle;
    }

    vos_gettimeofday(&cur_tim); 

    {
        vos_mutex_lock( g_DnlEntryEp.session.mutex );
        channel_report = g_DnlEntryEp.session.channel_status_report;
        vos_mutex_unlock( g_DnlEntryEp.session.mutex );
    }

    if ( (cur_tim.sec - session->last_report_time) >= repcycle 
        || session->media_status_report.cur_index > 0
        || channel_report )
    {
        vos_uint32_t req_seq = ++session->req_seq;
        if ( dnl_tp_tx_data( &session->tp, MSG_ID_DEV_STATUS_REPORT, MSG_TYPE_REQ, &req_seq, 0) != 0 )
        {
            DNL_ERROR_LOG(
                "entry_session_status_report error, sock_fd=%d, cur_tim=%d, last_reptim=%d, repcycle=%d\n", 
                session->tp.sock, 
                cur_tim.sec, 
                session->last_report_time, 
                repcycle );
            return -1;
        }

        DNL_TRACE_LOG(
            "send device status report success, sock_fd=%d, cur_tim=%d, last_reptim=%d, repcycle=%d\n", 
            session->tp.sock, 
            cur_tim.sec, 
            session->last_report_time, 
            repcycle );

        session->last_report_time = cur_tim.sec;

        return 1;
    }

    return 0;
}

static int entry_session_alarm_report(entry_session_t *session)
{
    vos_mutex_lock( g_DnlEntryEp.session.mutex );
    if( !vos_list_empty(&g_DnlEntryEp.session.alarm_list) )
    {
        vos_mutex_unlock( g_DnlEntryEp.session.mutex );

        {
            vos_uint32_t req_seq = ++session->req_seq;
            if ( dnl_tp_tx_data( &session->tp, MSG_ID_DEV_ALARM_REPORT, MSG_TYPE_REQ, &req_seq, 0) != 0 )
            {
                DNL_ERROR_LOG(
                    "entry_session_alarm_report error, sock_fd=%d\n", 
                    session->tp.sock );
                return -1;
            }
        }
        
        return 1;
    }
    
    vos_mutex_unlock( g_DnlEntryEp.session.mutex );

    return 0;
}

static int entry_session_on_normal_run(void* arg)
{
    int rst = RS_KP;
    tp_tcp_state_e tcp_state;
    entry_session_t* session = (entry_session_t*)arg;

    do 
    {
        //01. check tcp state
        tcp_state = dnl_transport_tcp_state( &session->tp );
        if ( tcp_state != en_tcp_state_connected )
        {
            DNL_ERROR_LOG(
                "error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d\n", 
                session->tp.sock, 
                tcp_state, 
                session->tp.tx.flag, 
                session->tp.rx.flag );
            rst = RS_NG;
            break;
        }

        //02. check tcp remained tx data
        if( dnl_transport_tx_writing(&session->tp) )
        {
            if( dnl_transport_send(&session->tp) < 0 )
            {
                DNL_DEBUG_LOG(
                    "stream_on_normal_run-->send remained data fail, sock_fd=%d, data_size=%d, sent_size=%d.\n", 
                    session->tp.sock, 
                    session->tp.tx.data_size, 
                    session->tp.tx.sent_size );
                rst = RS_NG;
                break;
            }

            break;
        }

        //03. handle rx data
        {
            int ret = entry_on_recv_msg();
            if( ret == RS_NG )
            {
                DNL_ERROR_LOG(
                    "handle recv msg, sock_fd=%d, tp_state=%d, msg_id=%u, msg_type=%u.\n", 
                    session->tp.sock, 
                    tcp_state, 
                    session->recv_msg.header.msg_id, 
                    session->recv_msg.header.msg_type );
                rst = RS_NG;
                break;
            }
            else if( ret == RS_KP )
            {
                break; //wait recv msg handle complete!
            }

            ret = dnl_tp_rx_data(&session->tp, 0, 0, 0 );
            if( ret == en_tp_rx_err )
            {
                DNL_ERROR_LOG(
                    "receive error, sock_fd=%d, tp_state=%d, tx_flag=%d, rx_flag=%d.\n", 
                    session->tp.sock, 
                    tcp_state, 
                    session->tp.tx.flag, 
                    session->tp.rx.flag );
                rst = RS_NG;
                break;
            }
        }

        //04. status report and media dispactch
        if( dnl_transport_tx_enable(&session->tp) )
        {
            // handle_fail: -1, handle_nothing:0, handle_some:1
            int ret = entry_session_status_report(session);
            if ( ret < 0){ rst = RS_NG; break; }
            else if( ret >0 ){ break; }

            ret = entry_session_alarm_report(session);
            if ( ret < 0){ rst = RS_NG; break; }
            else if( ret >0 ){ break; }

        }
    } while (0);

    return rst;
}

static int entry_session_on_error(void* arg)
{
    int ret = RS_OK;
    entry_session_t* session = (entry_session_t*)arg;

    DNL_DEBUG_LOG("entry_session_on_error-->sock_fd=%d, tx_flag=%d, rx_flag=%d.\n", 
        session->tp.sock, session->tp.tx.flag, session->tp.rx.flag);

    Clear_DH_conn_status( session->tp.sock );
    dnl_transport_close(&session->tp);

    session->in_main_proc = FALSE;

    session->report_cycle = 0;
    session->last_report_time = 0;

    session->sub_seq = 0;
    session->sub_start_time = 0;

    session->recv_msg.has_recv_msg_flag = false;   //clear
    session->media_status_report.cur_index = 0;    //clear

    if( session->snap_task.snap_task_tbl )
    {
        int i = 0;
        for( ; i<session->snap_task.max_channel_num; ++i )
        {
            entry_snap_i_t* snap_task_i = &session->snap_task.snap_task_tbl[i];
            snap_task_i->snaping_flag = FALSE;
            snap_task_i->pic_size = 0;
            snap_task_i->pic_sent_size = 0;
        }
    }

    if( vos_get_system_tick() - session->start_time < 500 )
    {
        session->retry_cnt++;
        if(session->retry_cnt > 5)
        {
            session->retry_cnt = 0;
            ret = RS_NG;
        }
    }
    else
    {
        session->retry_cnt = 0;
    }

    return ret;
}

int entry_build_msg_device_login_req(void *msg_buf, vos_size_t buf_len)
{
    int i, ret = 0;
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;

    DeviceLoginReq req;
    memset(&req, 0x0, sizeof(req));

    do 
    {
        if ( !msg_buf || buf_len<sizeof(MsgHeader) )
        {
            ret = -1;
            break;
        }

        head_pos = (char*)msg_buf;
        body_pos = head_pos + sizeof(MsgHeader);

        {
            //0x01
            req.mask = 0x01;
            strncpy(req.device_id, g_DnlDevInfo.dev_id, MAX_DEV_ID_LEN);
            strcpy(req.version, "8888");
            req.dev_type = g_DnlDevInfo.dev_type;
            req.channel_num = g_DnlDevInfo.channel_num;

            req.channels = VOS_MALLOC_BLK_T(DevChannelInfo, g_DnlDevInfo.channel_num);
            if( !req.channels )
            {
                ret = -2;
                break;
            }

            {
                vos_mutex_lock(g_DnlDevInfoMutex);
                for(i=0; i<req.channel_num; i++)
                {
                    req.channels[i].channel_id = g_DnlDevInfo.channel_list[i].channel_index;
                    req.channels[i].channel_status = g_DnlDevInfo.channel_list[i].channel_status;
                    req.channels[i].has_ptz = g_DnlDevInfo.channel_list[i].has_ptz;
                    req.channels[i].stream_num = g_DnlDevInfo.channel_list[i].stream_num;
                    {
                        int j = 0;
                        for( ; j<g_DnlDevInfo.channel_list[i].stream_num; ++j )
                        {
                            req.channels[i].stream_list[j].stream_id = g_DnlDevInfo.channel_list[i].stream_list[j].stream_id;
                            req.channels[i].stream_list[j].video_height = g_DnlDevInfo.channel_list[i].stream_list[j].video_height;
                            req.channels[i].stream_list[j].video_width = g_DnlDevInfo.channel_list[i].stream_list[j].video_width;
                            req.channels[i].stream_list[j].video_codec.codec_fmt = g_DnlDevInfo.channel_list[i].stream_list[j].video_codec.codec_fmt;
                        }
                    }
                    memcpy(&req.channels[i].audio_codec, &g_DnlDevInfo.channel_list[i].adudo_codec, sizeof(Dev_Audio_Codec_Info_t));
                }
                vos_mutex_unlock(g_DnlDevInfoMutex);
            }
            
            memcpy(&req.token, &g_DnlEntryEp.session.token, sizeof(token_t));

            body_size = Pack_MsgDeviceLoginReq(body_pos, buf_len-sizeof(MsgHeader), &req);
            if ( body_size<0 )
            {
                ret = -3;
                break;
            }
        }

        msg_size = body_size + sizeof(MsgHeader);

        {
            MsgHeader header;
            header.msg_size = msg_size;
            header.msg_id = MSG_ID_DEV_LOGIN;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++g_DnlEntryEp.session.req_seq;

            if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
            {
                ret = -4;
                break;
            }
        }

        ret = msg_size;

    } while (0);
    
    if( req.channels )
    {
        VOS_FREE_T(req.channels);
    }

    return ret;
}

int entry_build_msg_ability_report_req(void *msg_buf, vos_size_t buf_len)
{
    vos_size_t msg_size=0, body_size = 0;
    char* head_pos;
    char* body_pos;

    if ( !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    {
        DeviceAbilityReportReq req;
        memset(&req, 0x0, sizeof(req));

        req.mask = 0x01;
        req.media_trans_type = 1; // 默认TCP

        // 暂不限制
        req.max_live_streams_per_ch = 0;
        req.max_playback_streams_per_ch = 0;
        req.max_playback_streams = 0;
        
        /*
        req.mask |= 0x02;
        req.disc_size = 0;
        req.disc_free_size = 0;
        */

        body_size = Pack_MsgDeviceAbilityReportReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            return -2;
        }
    }
    
    msg_size = body_size + sizeof(MsgHeader);

    {
        MsgHeader header;
        header.msg_size = msg_size;
        header.msg_id = MSG_ID_DEV_ABILITY_REPORT;
        header.msg_type = MSG_TYPE_REQ;
        header.msg_seq = ++g_DnlEntryEp.session.req_seq;

        if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
        {
            return -3;
        }
    }

    return msg_size;
}
int entry_build_msg_status_report_req(void *msg_buf, vos_size_t buf_len)
{
    int ret;
    vos_size_t msg_size=0, body_size = 0;
    char *head_pos, *body_pos;
    DeviceStatusReportReq req;

    if ( !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    do 
    {
        int i=0;
        vos_bool_t channel_report = false;
        
        memset(&req, 0x0, sizeof(req));

        {
            vos_mutex_lock( g_DnlEntryEp.session.mutex );
            channel_report = g_DnlEntryEp.session.channel_status_report;
            g_DnlEntryEp.session.channel_status_report = false; //clear
            vos_mutex_unlock( g_DnlEntryEp.session.mutex );
        }

        if(channel_report)
        {
            req.mask = 0x01;
            req.channel_num = g_DnlDevInfo.channel_num;

            req.channels = VOS_MALLOC_BLK_T(DevChannelInfo, g_DnlDevInfo.channel_num);
            if( !req.channels )
            {
                ret = -2;
                break;
            }

            vos_mutex_lock(g_DnlDevInfoMutex);
            for(i=0; i<req.channel_num; i++)
            {
                req.channels[i].channel_id = g_DnlDevInfo.channel_list[i].channel_index;
                req.channels[i].channel_status = g_DnlDevInfo.channel_list[i].channel_status;
                req.channels[i].has_ptz = g_DnlDevInfo.channel_list[i].has_ptz;
                req.channels[i].stream_num = g_DnlDevInfo.channel_list[i].stream_num;
                {
                    int j = 0;
                    for( ; j<g_DnlDevInfo.channel_list[i].stream_num; ++j )
                    {
                        req.channels[i].stream_list[j].stream_id = g_DnlDevInfo.channel_list[i].stream_list[j].stream_id;
                        req.channels[i].stream_list[j].video_height = g_DnlDevInfo.channel_list[i].stream_list[j].video_height;
                        req.channels[i].stream_list[j].video_width = g_DnlDevInfo.channel_list[i].stream_list[j].video_width;
                        req.channels[i].stream_list[j].video_codec.codec_fmt = g_DnlDevInfo.channel_list[i].stream_list[j].video_codec.codec_fmt;
                    }
                }
                memcpy(&req.channels[i].audio_codec, &g_DnlDevInfo.channel_list[i].adudo_codec, sizeof(Dev_Audio_Codec_Info_t));
            }
            vos_mutex_unlock(g_DnlDevInfoMutex);
        }

        if( g_DnlEntryEp.session.last_report_time == 0 )
        {
            // at first time report status, report all stream media sessions!
            media_session_t* p = g_DnlStreamEp.live_session_list.next;
            for( ; p != &g_DnlStreamEp.live_session_list; p=p->next )
            {
                DeviceMediaSessionStatus media_status;
                media_session_t* media_session = p;
                if ( !media_session->running )
                {
                    continue;
                }

                strncpy( media_status.session_id, media_session->session_id, MAX_MEDIA_SESSION_ID_LEN);
                media_status.session_type = media_session->session_type;
                strncpy(media_status.device_id, g_DnlDevInfo.dev_id, MAX_DEV_ID_LEN );
                media_status.channel_id = media_session->channel_id;;
                media_status.stream_id = media_session->stream_id;

                if ( media_session->is_audio_open && media_session->is_video_open )
                {
                    media_status.session_media = 3;
                }
                else if (media_session->is_audio_open)
                {
                    media_status.session_media = 2;
                }
                else if (media_session->is_video_open)
                {
                    media_status.session_media = 1;
                }
                else
                {
                    media_status.session_media = 0;
                }

                if (media_session->status == en_stream_session_on_close)
                {
                    media_status.session_status = 0;
                }
                else if (media_session->status == en_stream_session_on_connecting)
                {
                    media_status.session_status = 1;
                }
                else
                {
                    media_status.session_status = 2;
                }

                {
                    addr_info* stream_serv_addr = &media_session->stream_svr_list[media_session->cur_stream_svr_idx];
                    strncpy( media_status.stream_addr.ip, stream_serv_addr->IP, MAX_IP_LEN);
                    media_status.stream_addr.port = stream_serv_addr->port;
                }

                entry_media_staus_chg( media_status );
            }
        }
        
        if( g_DnlEntryEp.session.media_status_report.cur_index > 0 )
        {
            req.mask |= 0x02;
            req.media_session_num = g_DnlEntryEp.session.media_status_report.cur_index;
            req.media_sessions = g_DnlEntryEp.session.media_status_report.status_list;

            g_DnlEntryEp.session.media_status_report.cur_index = 0; //reset index
        }

        body_size = Pack_MsgDeviceStatusReportReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            ret = -3;
            break;
        }

        msg_size = body_size + sizeof(MsgHeader);

        {
            MsgHeader header;
            header.msg_size = msg_size;
            header.msg_id = MSG_ID_DEV_STATUS_REPORT;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++g_DnlEntryEp.session.req_seq;

            if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
            {
                ret = -4;
            }
        }

        ret = msg_size;
    } while (0);

    if( req.channels )
    {
        VOS_FREE_T(req.channels);
    }

    return ret;
}

int entry_build_msg_alarm_report_req(void *msg_buf, vos_size_t buf_len)
{
    int ret;
    char *head_pos, *body_pos;
    DeviceAlarmReportReq req;
    memset(&req, 0x0, sizeof(req));

    if ( !msg_buf || buf_len<sizeof(MsgHeader) )
    {
        return -1;
    }

    head_pos = (char*)msg_buf;
    body_pos = head_pos + sizeof(MsgHeader);

    do 
    {
        vos_size_t msg_size=0, body_size = 0;
        entry_alarm_t* alarm_node;

        vos_mutex_lock( g_DnlEntryEp.session.mutex );
        alarm_node = (entry_alarm_t*)vos_list_pop_front( &g_DnlEntryEp.session.alarm_list );
        vos_mutex_unlock( g_DnlEntryEp.session.mutex );

        if( !alarm_node )
        {
            ret = -2;
            break;
        }

        req.mask = 0x01;
        strcpy(req.device_id, g_DnlDevInfo.dev_id);
        req.channel_id = alarm_node->channel_index;
        req.alarm_type = alarm_node->type;
        req.alarm_status = alarm_node->status;
        
        VOS_FREE_T(alarm_node);

        body_size = Pack_MsgDeviceAlarmReportReq(body_pos, buf_len-sizeof(MsgHeader), &req);
        if ( body_size<0 )
        {
            ret = -3;
            break;
        }

        msg_size = body_size + sizeof(MsgHeader);

        {
            MsgHeader header;
            header.msg_size = msg_size;
            header.msg_id = MSG_ID_DEV_ALARM_REPORT;
            header.msg_type = MSG_TYPE_REQ;
            header.msg_seq = ++g_DnlEntryEp.session.req_seq;

            if ( Pack_MsgHeader(head_pos, sizeof(MsgHeader), &header) < 0 )
            {
                ret = -4;
            }
        }

        ret = msg_size;
    } while (0);

    return ret;
}

static int entry_on_msg_media_open_req(entry_session_t *session, MsgHeader* header, DeviceMediaOpenReq* req)
{
    int ret = RS_KP;
    vos_bool_t send_resp_flag = false;
    DeviceMediaOpenResp resp;

    switch ( session->recv_msg.recv_msg_seq )
    {
    case 0:
        {
            int i = 0;
            vos_in_addr in_addr;
            media_desc_t media_desc;
            memset(&media_desc, 0x0, sizeof(media_desc_t));
            strncpy(media_desc.session_id, req->session_id, MAX_MEDIA_SESSION_ID_LEN);
            media_desc.session_type = req->session_type;
            media_desc.channel_id = req->channel_id;
            media_desc.stream_id = req->stream_id;
            if ( req->session_media & 0x01 )
            {
                media_desc.is_video_open = true;
            }
            else if ( req->session_media & 0x02 )
            {
                media_desc.is_audio_open = true;
            }
            else if ( req->session_media & 0x03 )
            {
                media_desc.is_audio_open = true;
                media_desc.is_video_open = true;
            }

            if (req->mask&0x20)
            {
                media_desc.stream_svr_list_len = req->stream_num;
                for ( ; i < req->stream_num; i++)
                {
                    strncpy(media_desc.stream_svr_list[i].IP, req->stream_servers[i].ip, MAX_IP_LEN+1);
                    media_desc.stream_svr_list[i].port = req->stream_servers[i].port;

                    in_addr = vos_inet_addr2(req->stream_servers[i].ip);
                    media_desc.stream_svr_list[i].sin_addr = in_addr.s_addr;
                }
            }

            if ( stream_media_session_open(media_desc) == 0 )
            {
                session->recv_msg.recv_msg_seq ++;
                session->recv_msg.recv_msg_time = vos_get_system_tick();

                //notify app live open
                if( req->session_type == MEDIA_SESSION_TYPE_LIVE )
                {
                    vos_uint32_t s_tick, e_tick, delt_tick;
                    Dev_Cmd_Param_t cmd_param;
                    cmd_param.cmd_type = EN_CMD_LIVE_OPEN;
                    cmd_param.channel_index = req->channel_id;
                    cmd_param.cmd_args[0] = req->stream_id;

                    s_tick = vos_get_system_tick();
                    g_DnlAppCb_func(g_DnlAppCb_userdata, &cmd_param);
                    e_tick = vos_get_system_tick();
                    delt_tick = e_tick - s_tick;
                    if( delt_tick > 200 )
                    {
                        DNL_WARN_LOG(
                            "app cb func handle too long, (%d, %d, %d), (%u, %u, %u)!\n",
                            (int)cmd_param.cmd_type, (int)cmd_param.channel_index, (int)cmd_param.cmd_args[0], 
                            s_tick, e_tick, delt_tick);
                    }
                }
            }
            else
            {
                memset(&resp, 0x0, sizeof(resp));
                resp.resp_code = EN_DEV_ERR_CREATE_STREAM_FAIL;
                send_resp_flag = TRUE;
                ret = RS_NG;
            }
        }
        break;
    case 1:
        {
            vos_bool_t is_media_session_open = stream_media_session_is_open( req->session_id );
            if( is_media_session_open )
            {
                memset(&resp, 0x0, sizeof(resp));
                resp.resp_code = EN_SUCCESS;
                send_resp_flag = TRUE;
            }
            else
            {
                vos_uint32_t curr_time = vos_get_system_tick();
                if ( curr_time - session->recv_msg.recv_msg_time > 5*1000)
                {
                    memset(&resp, 0x0, sizeof(resp));
                    resp.resp_code = EN_DEV_ERR_CONNECT_STREAM_FAIL;
                    send_resp_flag = TRUE;
                }
            }
        }
        break;
    default:
        {
            session->recv_msg.recv_msg_seq = 0;
            session->recv_msg.recv_msg_time = 0;
        }
        break;
    }

    if( send_resp_flag )
    {
        resp.mask = 0x01;
        strncpy(resp.device_id, req->device_id, MAX_DEV_ID_LEN);
        resp.channel_id = req->channel_id;
        resp.stream_id = req->stream_id;
        strncpy(resp.session_id, req->session_id, MAX_MEDIA_SESSION_ID_LEN);
        strncpy(resp.stream_server.ip, req->stream_servers[0].ip, MAX_IP_LEN+1);
        resp.stream_server.port = req->stream_servers[0].port;
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = RS_NG;
        }
        else
        {
            ret = RS_OK;
        }
    }
    
    return ret;
}

static int entry_on_msg_snap_req(entry_session_t *session, MsgHeader* header, DeviceSnapReq* req)
{
    int ret = RS_KP;
    vos_bool_t send_resp_flag = false;
    DeviceSnapResp resp;
    entry_snap_i_t* snap_task = &session->snap_task.snap_task_tbl[req->channel_id-1];

    switch ( session->recv_msg.recv_msg_seq )
    {
    case 0:
        {
            if( !snap_task->snaping_flag )
            {
                //init snap info
                {
                    snap_task->snaping_flag = TRUE;
                    snap_task->pic_size = 0;
                    snap_task->pic_sent_size = 0;
                }
                
                //notify app snap 
                {
                    Dev_Cmd_Param_t cmd_param;
                    cmd_param.cmd_type = EN_CMD_SNAP;
                    cmd_param.channel_index = req->channel_id;
                    g_DnlAppCb_func(g_DnlAppCb_userdata, &cmd_param);
                }

                //handle seq set
                session->recv_msg.recv_msg_seq ++;
                session->recv_msg.recv_msg_time = vos_get_system_tick();
            }
            else
            {
                resp.resp_code = EN_DEV_ERR_IN_BUSY;
                send_resp_flag = TRUE;
                ret = RS_OK;
            }
        }
        break;
    case 1:
        {
            if( snap_task->pic_size > 0 )
            {
                memset(&resp, 0x0, sizeof(resp));
                resp.resp_code = EN_SUCCESS;

                resp.mask = 0x01;
                {
                    strcpy(resp.device_id, req->device_id);
                    resp.channel_id = req->channel_id;
                    strcpy(resp.pic_fmt, entry_pic_fmt_str[snap_task->pic_fmt]);
                    resp.pic_size = snap_task->pic_size;
                }

                resp.mask |= 0x02;
                {
                    vos_uint32_t remain_size = snap_task->pic_size - snap_task->pic_sent_size;
                    vos_uint32_t send_size = (remain_size>ENTRY_PIC_SNED_SIZE)?ENTRY_PIC_SNED_SIZE:remain_size;
                    resp.offset = snap_task->pic_sent_size;
                    resp.data_size = send_size;
                    memcpy(resp.datas, snap_task->pic_buffer+snap_task->pic_sent_size, send_size);

                    snap_task->pic_sent_size += send_size;
                }

                send_resp_flag = TRUE;
                if( snap_task->pic_sent_size >= snap_task->pic_size )
                {
                    snap_task->snaping_flag = FALSE; //handle snap end, clear flag
                    ret = RS_OK;
                }
            }
            else
            {
                vos_uint32_t curr_time = vos_get_system_tick();
                if ( curr_time - session->recv_msg.recv_msg_time > 2*1000)
                {
                    memset(&resp, 0x0, sizeof(resp));
                    resp.resp_code = EN_DEV_ERR_TIMEOUT;
                    resp.mask = 0x01;
                    {
                        strcpy(resp.device_id, req->device_id);
                        resp.channel_id = req->channel_id;
                    }

                    send_resp_flag = TRUE;
                    ret = RS_OK;

                    snap_task->snaping_flag = FALSE; //handle snap end, clear flag
                }
            }
        }
        break;
    default:
        {
            session->recv_msg.recv_msg_seq = 0;
            session->recv_msg.recv_msg_time = 0;
        }
        break;
    }

    if( send_resp_flag )
    {
        if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
        {
            ret = RS_NG;
        }
    }

    return ret;
}

static int entry_on_msg_device_ctrl_req(entry_session_t *session, MsgHeader* header, DeviceCtrlReq* req)
{
    int ret = RS_OK;
    DeviceCtrlResp resp;
    memset(&resp, 0x0, sizeof(resp));
    do
    {
        Dev_Cmd_Param_t cmd_param;
        memset(&cmd_param, 0x0, sizeof(cmd_param));

        if( req->cmd_type == DEVICE_CMD_PTZ )
        {
            cmd_param.cmd_type = EN_CMD_PTZ;
        }
        else if( req->cmd_type == DEVICE_CMD_MGR_UPDATE )
        {
            cmd_param.cmd_type = EN_CMD_MGR_UPDATE;
        }
        else
        {
            cmd_param.cmd_type = req->cmd_type;
        }
        cmd_param.channel_index = req->channel_id;
        memcpy(cmd_param.cmd_args, req->cmd_datas, req->cmd_data_size);
        g_DnlAppCb_func(g_DnlAppCb_userdata, &cmd_param);
        
        resp.resp_code = EN_SUCCESS;

    } while (0);

    //发送响应
    if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
    {
        ret = RS_NG;
    }
    else
    {
        ret = RS_OK;
    }

    return ret;
}

static int entry_on_msg_media_close_req(entry_session_t *session, MsgHeader* header, DeviceMediaCloseReq* req)
{
    int ret = RS_OK;
    DeviceMediaCloseResp resp;

    do 
    {
        memset(&resp, 0x0, sizeof(resp));

        if (!req->mask&0x01)
        {
            resp.resp_code = EN_ERR_MSG_PARSER_FAIL;
            break;
        }

        if (0 != stream_media_session_close(req->session_id))
        {
            resp.resp_code = EN_ERR_NOT_SUPPORT;
            break;
        }

        resp.resp_code = EN_SUCCESS;

    } while (0);

    //发送响应
    if ( dnl_tp_tx_data(&session->tp, header->msg_id, MSG_TYPE_RESP, &header->msg_seq, &resp) < 0 )
    {
        ret = RS_NG;
    }
    else
    {
        ret = RS_OK;
    }

    return ret;
}

int entry_on_msg_device_login_resp(DeviceLoginResp* resp)
{
    int ret = RS_NG;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = RS_NG;
            break;
        }

        if (!resp->mask&0x01)
        {
            ret = RS_NG;
            break;
        }

        {
            //vos_in_addr t_in_addr;
            //g_DnlDevInfo.public_ip = vos_inet_aton(resp->public_ip, &t_in_addr);
            //g_DnlDevInfo.public_port = resp->public_port;

            ret = RS_OK;
        }
        
    } while (0);

    return ret;
}

static int entry_on_msg_device_ability_report_resp(entry_session_t *session, DeviceAbilityReportResp* resp)
{
    int ret = RS_NG;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = RS_NG;
            break;
        }

        ret = RS_OK;
    } while (0);

    return ret;
}

static int entry_on_msg_device_status_report_resp(entry_session_t *session, DeviceStatusReportResp* resp)
{
    int ret = RS_NG;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = RS_NG;
            break;
        }

        if (resp->mask&0x01)
        {
            g_DnlEntryEp.session.report_cycle = resp->expected_cycle;
        }
        ret = RS_OK;
    } while (0);

    return ret;
}

static int entry_on_msg_device_alarm_report_resp(entry_session_t *session, DeviceAlarmReportResp* resp)
{
    int ret = RS_NG;
    do 
    {
        if (resp->resp_code != EN_SUCCESS)
        {
            ret = RS_NG;
            break;
        }
        ret = RS_OK;
    } while (0);

    return ret;
}