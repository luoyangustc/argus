#define __DEVICE_SDK_C__

#include "vos_types.h"
#include "DeviceSDK.h"
#include "dnl_dev.h"
#include "dnl_ctrl.h"
#include "dnl_log.h"
#include "dnl_stream.h"
#include "dnl_entry.h"

static vos_uint32_t g_sys_start_time = 0;

int Dev_Sdk_Init(Entry_Serv_t* entry_serv, Dev_Info_t *dev_info)
{
	vos_time_val tv;
    int i = 0;

    if( !entry_serv || !dev_info )
    {
        return -1;
    }

    g_DnlDevInfoMutex = NULL;
    if( vos_mutex_create_recursive( NULL, &g_DnlDevInfoMutex ) != VOS_SUCCESS )
    {
        return -2;
    }

    /* initialize global device info to zero */
    //memset( &g_DnlDevInfo, 0x0, sizeof(g_DnlDevInfo) );

    /* save entry serv address info */
    memcpy( &g_DnlEntryAddr, entry_serv, sizeof(Entry_Serv_t) );

    /* save device info */
	memcpy( &g_DnlDevInfo, dev_info, sizeof(g_DnlDevInfo) );
    //g_DnlDevInfo.dev_type = dev_info->dev_type;
    //memcpy( &g_DnlDevInfo.oem_info, &dev_info->oem_info, sizeof(Dev_OEM_Info_t) );
    //memcpy( &g_DnlDevInfo.attr, &dev_info->attr, sizeof(Dev_Attribute_t) );
    //g_DnlDevInfo.channel_num = dev_info->channel_num;
    g_DnlDevInfo.channel_list = VOS_MALLOC_BLK_T(Dev_Channel_Info_t, dev_info->channel_num);
    if( !g_DnlDevInfo.channel_list )
    {
        return -2;
    }

    for( i=0; i<dev_info->channel_num; ++i )
    {
        g_DnlDevInfo.channel_list[i].channel_index = dev_info->channel_list[i].channel_index;
        g_DnlDevInfo.channel_list[i].channel_status = dev_info->channel_list[i].channel_status;
        g_DnlDevInfo.channel_list[i].has_ptz = dev_info->channel_list[i].has_ptz;
        g_DnlDevInfo.channel_list[i].stream_num = dev_info->channel_list[i].stream_num;
        if( dev_info->channel_list[i].stream_num > 0 )
        {
            g_DnlDevInfo.channel_list[i].stream_list = VOS_MALLOC_BLK_T(Dev_Stream_Info_t, dev_info->channel_list[i].stream_num);
            if( !g_DnlDevInfo.channel_list[i].stream_list )
            {
                return -3;
            }
            memcpy( g_DnlDevInfo.channel_list[i].stream_list,
                dev_info->channel_list[i].stream_list, 
                sizeof(Dev_Stream_Info_t) * dev_info->channel_list[i].stream_num );
        }
        else
        {
            g_DnlDevInfo.channel_list[i].stream_list = NULL;
        }

        memcpy( &g_DnlDevInfo.channel_list[i].adudo_codec, 
            &dev_info->channel_list[i].adudo_codec, 
            sizeof(Dev_Audio_Codec_Info_t) );
    }
	
	if ( dnl_init() != VOS_SUCCESS )
	{
		return -1;
	}

	vos_gettimeofday(&tv);
	g_sys_start_time = tv.sec;
	
	return 0;
}

void Dev_Sdk_Uninit(void)
{
    dnl_destroy();
}

void Dev_Sdk_Set_CB( Dev_Cmd_Cb_Func cb_func, void* user_data )
{
    entry_app_cb_set( cb_func, user_data );
}

int Dev_Sdk_Stream_Frame_Report( Dev_Stream_Frame_t *frame )
{
	unsigned int ret=0;
	media_session_t *pmedia_session;

	if ( !frame )
	{
		DNL_DEBUG_LOG("pEvent is nil\n!");
		return -1;
	}

	pmedia_session = stream_get_media_session( MEDIA_SESSION_TYPE_LIVE, frame->channel_index, frame->stream_id );
	if ( !pmedia_session )
	{
		return -1;
	}
	
	ret = stream_media_event_report(pmedia_session, frame);

	return  ret;	//返回缓冲区使用数量
}

void Dev_Sdk_Alarm_Report(Dev_Alarm_t alarm)
{
    entry_alarm_t* alarm_node = VOS_MALLOC_T(entry_alarm_t);
    if( !alarm_node )
    {
        return;
    }
    alarm_node->channel_index = alarm.channel_index;
    alarm_node->type = alarm.type;
    alarm_node->status = alarm.status;

    vos_mutex_lock( g_DnlEntryEp.session.mutex );
    vos_list_push_back( &g_DnlEntryEp.session.alarm_list, alarm_node );
    vos_mutex_unlock( g_DnlEntryEp.session.mutex );
}

sdk_uint64 Dev_Sdk_Get_Timestamp(void)
{
	return vos_get_system_tick();
}

char* Dev_Sdk_Get_Device_ID(void)
{
    return g_DnlDevInfo.dev_id;
}

int Dev_Sdk_Channel_Status_Report(sdk_uint16 channel_index, Dev_Channel_Info_t* status)
{
    if( channel_index == 0 
        || channel_index > g_DnlDevInfo.channel_num  
        || !status )
    {
        return -1;
    }

    {
        vos_mutex_lock(g_DnlDevInfoMutex);
        g_DnlDevInfo.channel_list[channel_index-1].channel_index = status->channel_index;
        g_DnlDevInfo.channel_list[channel_index-1].channel_status = status->channel_status;
        g_DnlDevInfo.channel_list[channel_index-1].has_ptz = status->has_ptz;

        if( g_DnlDevInfo.channel_list[channel_index-1].stream_num != status->stream_num )
        {
            if( g_DnlDevInfo.channel_list[channel_index-1].stream_list )
            {
                VOS_FREE_T(g_DnlDevInfo.channel_list[channel_index-1].stream_list);
                g_DnlDevInfo.channel_list[channel_index-1].stream_list = NULL;
            }

            if( status->stream_num > 0 )
            {
                g_DnlDevInfo.channel_list[channel_index-1].stream_list = VOS_MALLOC_BLK_T(Dev_Stream_Info_t, status->stream_num);
                if( !g_DnlDevInfo.channel_list[channel_index-1].stream_list )
                {
                    return -2;
                }
                memcpy( g_DnlDevInfo.channel_list[channel_index-1].stream_list, 
                    status->stream_list, 
                    sizeof(Dev_Stream_Info_t) * status->stream_num );
            }

            g_DnlDevInfo.channel_list[channel_index-1].stream_num = status->stream_num;
        }

        memcpy( &g_DnlDevInfo.channel_list[channel_index-1].adudo_codec, 
            &(status->adudo_codec), 
            sizeof(Dev_Audio_Codec_Info_t) );

        vos_mutex_unlock(g_DnlDevInfoMutex);
    }
    
    {
        vos_mutex_lock( g_DnlEntryEp.session.mutex );
        g_DnlEntryEp.session.channel_status_report = true;
        vos_mutex_unlock( g_DnlEntryEp.session.mutex );
    }
    
    return 0;
}

int Dev_Sdk_Snap_Picture_Report(sdk_uint16 channel_index, EN_PIC_FMT_TYPE pic_fmt, sdk_uint8 *pic_data, sdk_uint32 pic_size )
{
    if ( channel_index == 0 || channel_index > g_DnlDevInfo.channel_num )
    {
        return -1;
    }

    if ( !g_DnlEntryEp.session.snap_task.snap_task_tbl )
    {
        return -2;
    }

    if( pic_fmt != EN_PIC_FMT_BMP && pic_fmt != EN_PIC_FMT_JPEG )
    {
        return -3;
    }

    vos_mutex_lock( g_DnlEntryEp.session.mutex );
    {
        entry_snap_i_t* snap_task = &g_DnlEntryEp.session.snap_task.snap_task_tbl[channel_index-1];
        if ( snap_task->pic_buffer_size < pic_size )
        {
            vos_uint32_t new_buffer_size = pic_size + pic_size/10;

            if ( snap_task->pic_buffer )
            {
                VOS_FREE_T(snap_task->pic_buffer);
                snap_task->pic_buffer = NULL;
                snap_task->pic_buffer_size = 0;
            }

            snap_task->pic_buffer = VOS_MALLOC_BLK_T(vos_uint8_t, new_buffer_size);
            if(!snap_task->pic_buffer)
            {
                return -4;
            }
            snap_task->pic_buffer_size = new_buffer_size;
        }
        memcpy(snap_task->pic_buffer, pic_data, pic_size);
        snap_task->pic_sent_size = 0;
        snap_task->pic_fmt = pic_fmt;
        snap_task->pic_size = pic_size;
    }
    vos_mutex_unlock( g_DnlEntryEp.session.mutex );

    return 0;
}
