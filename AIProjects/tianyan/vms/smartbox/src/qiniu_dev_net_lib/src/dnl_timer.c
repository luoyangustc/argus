#define __DNL_TIMER_C__

#include "dnl_entry.h"
#include "dnl_timer.h"
#include "dnl_ctrl.h"

#include "dnl_log.h"
#if (OS_UCOS_II == 1)
#include "mytimer.h"
#endif


#if 0
typedef enum timer_id
{
	TIMER_ID_10MS = 1,
	TIMER_ID_1S
}timer_id;

typedef struct _dnl_timer
{
	vos_timer_entry timer_10ms;
	vos_timer_entry timer_1s;
}dnl_timer;

static dnl_timer g_DnlTimer;

static void start_10ms_timer();
static void start_1s_timer();

static void dnl_timer_cb(vos_timer_heap_t *timer_heap, struct vos_timer_entry *entry)
{
	vos_time_val tv;
	vos_gettimeofday(&tv);
	if (entry->id == TIMER_ID_10MS) 
	{
		//printf("%d.%06d----->timer out for 10ms!\n", tv.sec,tv.msec);
	}
	else if (entry->id == TIMER_ID_1S) 
	{
		entry_1s_timeout();
		dnl_log_1s_timeout();
		//printf("%d.%06d----->timer out for 1s!\n", tv.sec,tv.msec);
	}
}


static void start_10ms_timer()
{
	vos_time_val delay = {0};

	vos_timer_entry_init(&g_DnlTimer.timer_10ms,
		TIMER_ID_10MS,
		0,
		dnl_timer_cb);

	delay.sec = 0;
	delay.msec = 10; 
	vos_timer_heap_schedule(g_DnlTimerHeap, &g_DnlTimer.timer_10ms, &delay, &delay);
}

static void start_1s_timer()
{
	vos_time_val delay = {0};

	vos_timer_entry_init(&g_DnlTimer.timer_1s,
		TIMER_ID_1S,
		0,
		dnl_timer_cb);

	delay.sec = 1;
	delay.msec = 0; 
	vos_timer_heap_schedule(g_DnlTimerHeap, &g_DnlTimer.timer_1s, &delay, &delay);

	
}

void start_timer()
{
	start_10ms_timer();

	start_1s_timer();

}

void stop_timer()
{
	if (g_DnlTimer.timer_10ms.id != 0) 
	{
		vos_timer_heap_cancel(g_DnlTimerHeap, &g_DnlTimer.timer_10ms);
		g_DnlTimer.timer_10ms.id  = 0;	
	}

	if (g_DnlTimer.timer_1s.id != 0) 
	{
		vos_timer_heap_cancel(g_DnlTimerHeap, &g_DnlTimer.timer_1s);
		g_DnlTimer.timer_1s.id  = 0;	
	}
}
#else

#define MYTIMER_MAX_NUM 5

typedef void(*mytimer_callback)(void *param);
typedef struct
{
    mytimer_callback  m_pTimerFun;  //!< 定时器回调函数指针
    void        *m_pDat;            //!< 定时器回调函数参数

    vos_uint64_t    m_dwCallTime;   //!< 下一次调用时间
    vos_uint64_t    m_dwOldTime;    //!< 上一次调用时间
    unsigned int    m_Period;       //!< 时间周期
    unsigned int    m_Loop;         //!< 是否循环

    int m_bActive;      //!< 定时器是否已激活
    int m_bCalled;      //!< 定时器回调函数已经被调用过
}mytimer_t, *MYTIMER_HANDLE;

static MYTIMER_HANDLE g_DnlTimer;
mytimer_t g_mytimer_table[MYTIMER_MAX_NUM];
vos_mutex_t *mytimer_mutex;

#define MYTIMER_LOCK()			vos_mutex_lock(mytimer_mutex)
#define MYTIMER_UNLOCK()		vos_mutex_unlock(mytimer_mutex)



/*! 输入当前的时间，检查是否到时间，如果到则执行原定回调函数
\param handle 定时器句柄
\param dwCurTime 当前时间
\return 无 */
void mytimer_run(MYTIMER_HANDLE handle, vos_uint64_t dwCurTime)
{
    MYTIMER_HANDLE ptimer = handle;
    vos_uint64_t slap_time;

    if (!ptimer->m_bActive || (ptimer->m_Loop == 0 && ptimer->m_bCalled))
    {
        return;
    }

    if ((ptimer->m_dwOldTime <= dwCurTime))
        slap_time = dwCurTime - ptimer->m_dwOldTime;
    else
        slap_time = ptimer->m_dwOldTime - dwCurTime;    // this is not good, you can fix me

    if ((ptimer->m_Loop != 0) && (slap_time >= (int)ptimer->m_Period))
    {
        ptimer->m_dwCallTime = dwCurTime;
    }

    if ((dwCurTime >= ptimer->m_dwCallTime)
        || (dwCurTime < ptimer->m_dwOldTime))
    {
        ptimer->m_dwOldTime = dwCurTime;
        ptimer->m_dwCallTime += ptimer->m_Period;
        if (ptimer->m_dwCallTime < ptimer->m_dwOldTime)
        {
            ptimer->m_dwOldTime = 0;    // this is not good, you can fix me
        }

        ptimer->m_bCalled = 1;

#if 1   // to avoid net and timer enter dead lock state, this is not good method, fix me please.
#if NO_SYS //by PeterJiang for socket.
        extern void net_time_tick(void *param);
        if (net_time_tick == ptimer->m_pTimerFun)
        {
            MYTIMER_UNLOCK();
            (ptimer->m_pTimerFun)(ptimer->m_pDat);
            MYTIMER_LOCK();
        }
        else
#endif
        {
            (ptimer->m_pTimerFun)(ptimer->m_pDat);
        }
#else
        (ptimer->m_pTimerFun)(ptimer->m_pDat);
#endif
    }
}

static vos_thread_ret_t mytimer_task(void *parg)
{
    //unsigned char error;
    int i;

    while (1)
    {
#if MYTIMER_USE_SEM
        OSSemPend(mytimer_sem, 0, &error);
#else
        vos_thread_sleep(10);
#endif

        // lock
        MYTIMER_LOCK();

        for (i = 0; i < MYTIMER_MAX_NUM; i++)
        {
            if (g_mytimer_table[i].m_pTimerFun != NULL)
            {
                mytimer_run(&g_mytimer_table[i], vos_get_system_msec());
            }
        }

        // unlock
        MYTIMER_UNLOCK();
    }

    return 0;
}

int mytimer_init(void)
{
    vos_status_t rc;
    memset(&g_mytimer_table, 0, sizeof(g_mytimer_table));
    vos_mutex_create("time_mutex", 0, &mytimer_mutex);
    //mytimer_mutex = MYTIMER_CREATE();//OSSemCreate(1);
    if (NULL == mytimer_mutex)
    {
        uprintf("mytimer_init: OSSemCreate for mutex fail!\n");
        return -1;
    }

#if MYTIMER_USE_SEM
    mytimer_sem = OSSemCreate(0);
    if (NULL == mytimer_sem)
    {
        uprintf("mytimer_init: OSSemCreate for sem fail!\n");
        return -1;
    }
#endif

    // create task
    //vos_status_t rc;
    rc = vos_thread_create(
        "timer",
        mytimer_task,
        NULL,
        NULL,
        ENTRY_TASK_STACK_SIZE,
        0,
        DNL_ENTRY_TASK_PRIO,
        &g_DnlThreadID[DNL_THD_IDX_TIMER]);
    if (rc != VOS_SUCCESS)
    {
        DNL_ERROR_LOG("create entry_thread failed, %d!\n", rc);
        return -1;
    }

    //	memset(timerTaskStack, 0xff, sizeof(timerTaskStack));
    //	OSTaskCreate2(mytimer_task, NULL, &timerTaskStack[TIMER_TASK_STACK_SIZE - 1], TIMER_TASK_PRIO, TIMER_TASK_STACK_SIZE);
    //#if OS_TASK_NAME_SIZE > 1
    //	OSTaskNameSet(TIMER_TASK_PRIO, (INT8U *)"timer", NULL);
    //#endif

    return 0;
}


/*! 创建定时器
\param pTimerFun 定时器到时的回调函数
\param loop 是否循环
\param Period 定时器的周期，如果为0，则表示该定时器是一次性的，单位是毫秒
\param pDat 回调函数的参数，缺省值为NULL
\return 成功开启: 定时器句柄, 失败: NULL
\sa mytimer_destroy */
MYTIMER_HANDLE mytimer_create(mytimer_callback pTimerFun, unsigned int loop, unsigned int Period, void *pDat)
{
    MYTIMER_HANDLE ptimer = NULL;
    int i;
    //unsigned char error;

    if (pTimerFun == (mytimer_callback)NULL)
    {
        return ptimer;
    }

    MYTIMER_LOCK();

    do
    {
        for (i = 0; i < MYTIMER_MAX_NUM; i++)
        {
            if (g_mytimer_table[i].m_pTimerFun == NULL)
            {
                ptimer = &g_mytimer_table[i];
            }
        }

        if (NULL != ptimer)
        {
            ptimer->m_pTimerFun = pTimerFun;
            ptimer->m_pDat = pDat;
            ptimer->m_dwCallTime = vos_get_system_msec();
            ptimer->m_dwOldTime = ptimer->m_dwCallTime;
            ptimer->m_dwCallTime += Period;
            ptimer->m_Period = Period;
            ptimer->m_bCalled = 0;
            ptimer->m_bActive = 0;
            ptimer->m_Loop = loop;
        }
    } while (0);

    MYTIMER_UNLOCK();

    return ptimer;
}

/*! 开启定时器
\param handle 定时器句柄
\return 成功开启: 0, 失败: -1
\sa mytimer_stop */
int mytimer_start(MYTIMER_HANDLE ptimer)
{
    //int i;
    //unsigned char error;

    MYTIMER_LOCK();

    do
    {
        if (NULL != ptimer)
        {
            ptimer->m_dwCallTime = vos_get_system_msec();
            ptimer->m_dwOldTime = ptimer->m_dwCallTime;
            ptimer->m_dwCallTime += ptimer->m_Period;
            ptimer->m_bCalled = 0;
            ptimer->m_bActive = 1;
        }
    } while (0);

    MYTIMER_UNLOCK();

    return 0;
}


/*! 重新设置定时器
\param handle 定时器句柄
\param Period 定时器的周期，单位是毫秒
\return 成功-0 失败-1
\sa mytimer_stop */
int mytimer_reset(MYTIMER_HANDLE ptimer, unsigned int Period)
{
    //int i;
    //unsigned char error;

    MYTIMER_LOCK();

    do
    {
        if (NULL != ptimer)
        {
            ptimer->m_dwCallTime = vos_get_system_msec();
            ptimer->m_dwOldTime = ptimer->m_dwCallTime;
            ptimer->m_dwCallTime += Period;
            ptimer->m_Period = Period;
            ptimer->m_bCalled = 0;
        }
    } while (0);

    MYTIMER_UNLOCK();

    return 0;
}

/*! 关闭定时器
\param handle 定时器句柄
\return 成功: 0, 失败: -1
\sa mytimer_start */
int mytimer_stop(MYTIMER_HANDLE handle)
{
    int i;
    //unsigned char error;

    MYTIMER_LOCK();
    for (i = 0; i < MYTIMER_MAX_NUM; i++)
    {
        if (&g_mytimer_table[i] == handle)
        {
            g_mytimer_table[i].m_bActive = 0;
            break;
        }
    }
    MYTIMER_UNLOCK();

    return (i < MYTIMER_MAX_NUM) ? 0 : -1;
}

/*! 销毁定时器
\param handle 定时器句柄
\return 成功: 0, 失败: -1
\sa mytimer_create */
int mytimer_destroy(MYTIMER_HANDLE handle)
{
    int i;
    //unsigned char error;

    MYTIMER_LOCK();
    for (i = 0; i < MYTIMER_MAX_NUM; i++)
    {
        if (&g_mytimer_table[i] == handle)
        {
            g_mytimer_table[i].m_pTimerFun = NULL;
            g_mytimer_table[i].m_bActive = 0;
            break;
        }
    }
    MYTIMER_UNLOCK();

    return (i < MYTIMER_MAX_NUM) ? 0 : -1;
}

static void dnl_1s_timer_cb(void*param)
{
	entry_1s_timeout();
	//dnl_log_1s_timeout();
}

int dnl_start_timer()
{
    g_DnlTimer = mytimer_create(dnl_1s_timer_cb, 1, 1000, NULL);
    if( NULL != g_DnlTimer)
    {
        mytimer_start(g_DnlTimer);

        return 0;
    }

    return -1;
}

void dnl_stop_timer()
{
    mytimer_stop(g_DnlTimer);
}

#endif
