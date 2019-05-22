#define __DNL_CTRL_C__

#include "dnl_ctrl.h"
#include "dnl_entry.h"
#include "dnl_stream.h"
#include "dnl_event.h"
#include "dnl_dev.h"
//#include "dnl_probe.h"
#include "dnl_timer.h"
#include "dnl_log.h"


#define MAX_TIMER_COUNT 10
#define MAX_TIMED_OUT_ENTRIES 10

static vos_time_val gSystemStartTv;

#if (OS_UCOS_II == 1)
static OS_STK gEntryTaskStack[ENTRY_TASK_STACK_SIZE];
static OS_STK gStreamTaskStack[STREAM_TASK_STACK_SIZE];
static OS_STK gProbeTaskStack[OTHER_TASK_STACK_SIZE];
static OS_STK gAppCbTaskStack[OTHER_TASK_STACK_SIZE];
#endif

static vos_status_t dnl_sys_init()
{
    int i = 0;
    for(; i<DNL_THD_IDX_MAX; ++i)
    {
        g_DnlThreadID[i] = (vos_thread_t*)(0);
    }

	dnl_log_init();
	dnl_dev_init();
	dnl_entry_init();
	dnl_stream_init();

	vos_gettimeofday(&gSystemStartTv);

	DNL_INFO_LOG("dnl sys init success!\n");

	return VOS_SUCCESS;
}

static vos_status_t dnl_res_create()
{
    vos_status_t status = VOS_SUCCESS;
    return status;
}

vos_thread_ret_t worker_thread(vos_thread_arg_t arg)
{
    vos_uint32_t s_tick, e_tick, delt_tick;
    
    while(1)
    {
        s_tick = vos_get_system_tick();
        entry_run_proc(arg);

        stream_run_proc(arg);

        vos_thread_sleep(1);

        e_tick = vos_get_system_tick();
        delt_tick = e_tick - s_tick;
        if( delt_tick > 500 )
        {
            DNL_WARN_LOG(
                "work thread handle time too long,(%u, %u, %u)!\n",
                s_tick, e_tick, delt_tick);
        }
    }
}

static vos_status_t dnl_work_thread_create()
{
    vos_status_t rc;

    do 
    {
#if (OS_UCOS_II == 1)
        rc = vos_thread_create(
                    "entry_handle", 
                    &worker_thread,
                    NULL, 
                    &gEntryTaskStack[ENTRY_TASK_STACK_SIZE-1], 
                    ENTRY_TASK_STACK_SIZE,
                    0,
                    DNL_ENTRY_TASK_PRIO,
                    &g_DnlThreadID[DNL_THD_IDX_ENTRY]);
        if(rc != OS_NO_ERR) 
        {
            DNL_ERROR_LOG("create entry_thread failed, %d!\n", rc);
			break;
        }
        rc = vos_thread_create(
                    "probe_handle", 
                    probe_proc_run,
                    NULL, 
                    &gAppCbTaskStack[OTHER_TASK_STACK_SIZE-1], 
                    OTHER_TASK_STACK_SIZE,
                    0,
                    DNL_PROBE_TASK_PRIO,
                    &g_DnlThreadID[DNL_THD_IDX_PROBE]);
        if(rc != OS_NO_ERR) 
        {
            DNL_ERROR_LOG("create probe_proc_run failed, %d!\n", rc);
			break;
        }
#else

        rc = vos_thread_create(
            "entry_handle",
            &worker_thread,
            NULL,
            NULL,
            ENTRY_TASK_STACK_SIZE,
            0,
            DNL_ENTRY_TASK_PRIO,
            &g_DnlThreadID[DNL_THD_IDX_ENTRY]);
        if (rc != VOS_SUCCESS)
        {
            DNL_ERROR_LOG("create entry_thread failed, %d!\n", rc);
            break;
        }
#endif
    } while (0);

    return rc;
}

vos_status_t dnl_init(void)
{
	vos_status_t status;
	//vos_lock_t *lock = NULL;
	//int i =0;

	DNL_DEBUG_LOG("enter dnl_init().\n");

	if ( (status = vos_init() ) != VOS_SUCCESS )
	{
		uprintf(" vos_init() fail--->%d.\n", status);
		return status;
	}

	if ( (status = dnl_sys_init() ) != VOS_SUCCESS )
	{
		uprintf(" dnl_sys_init() fail--->%d.\n", status);
		return status;
	}
	
	if ( (status = dnl_res_create() ) != VOS_SUCCESS )
	{
		uprintf(" dnl_res_create() fail--->%d.\n", status);
		return status;
	}

	if ( (status = dnl_work_thread_create() ) != VOS_SUCCESS )
	{
		uprintf(" dnl_work_thread_create() fail--->%d.\n", status);
		return status;
	}

    mytimer_init();
	dnl_start_timer();
	
	DNL_DEBUG_LOG("exit enter dnl_init() ok.\n");
	return status;
}

void dnl_destroy(void)
{
    int i = 0;

	dnl_entry_final();

	dnl_stream_final();

	dnl_stop_timer();

    
    for(; i<DNL_THD_IDX_MAX; ++i)
    {
        if(g_DnlThreadID[i])
        {
            vos_thread_destroy(g_DnlThreadID[i]);
            g_DnlThreadID[i] = (vos_thread_t*)(0);
        }
    }
}


