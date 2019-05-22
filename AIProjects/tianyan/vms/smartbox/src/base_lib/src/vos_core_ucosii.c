#define __OS_CORE_C__

#include <stdio.h>
#include <string.h>

#include "includes.h"

#include "soc.h"
#include "gpio.h"
#include "uart.h"
#include "hardware_config.h"
#include "mytimer.h"
#include "base.h"
#include "udpserver.h"
#include "system.h"
#include "wifinet.h"
#include "myuart.h"
#include "rlock.h"
#include "wlan.h"
#include "rtc.h"


#include "types.h"
#include "os.h"

struct vos_thread_t
{
	char			obj_name[VOS_MAX_OBJ_NAME+1];
    vos_int32_t		thread_handle;
    vos_uint32_t   	thread_id;
};

typedef struct vos_mutex_t
{
    OS_EVENT    *mutex;
    char		obj_name[VOS_MAX_OBJ_NAME+1];
}vos_mutex_t;

typedef struct vos_sem_t
{
	OS_EVENT	*sem;
    char		obj_name[VOS_MAX_OBJ_NAME+1];
}vos_sem_t;


vos_status_t vos_mutex_create(const char *name, vos_mutex_t **ptr_mutex)
{
	if( NULL == ptr_mutex)
	{
		return -1;
	}

	*ptr_mutex = VOS_MALLOC_T(vos_mutex_t);
	if( NULL == (*ptr_mutex) )
	{
        return -1;
	}

	strncpy((*ptr_mutex)->obj_name, name, VOS_MAX_OBJ_NAME);
	

	(*ptr_mutex)->mutex = OSSemCreate(1);
    if(NULL == (*ptr_mutex)->mutex)
    {
        return -1;
    }

    /*
	(*ptr_mutex)->mutex = rLock_Create();
	if( NULL == ((*ptr_mutex)->mutex) )
	{
		return -1;
	}*/
	
	return VOS_SUCCESS;
}

vos_status_t vos_mutex_lock(vos_mutex_t *mutex)
{
    vos_uint8_t error;
    if(!mutex)
    {
        return -1;
    }

    /*
	rLock_Lock(mutex->mutex);
	*/

	OSSemPend(mutex->mutex, 0, &error);

	if(error != OS_NO_ERR)
	{
        return -1;
	}
	
	return VOS_SUCCESS;
}

vos_status_t vos_mutex_unlock(vos_mutex_t *mutex)
{
    if(!mutex)
    {
        return -1;
    }

    /*
	rLock_Unlock(mutex->mutex);
	*/

	OSSemPost(mutex->mutex);
	
	return VOS_SUCCESS;
}

vos_status_t vos_mutex_destroy(vos_mutex_t *mutex)
{
    if(!mutex)
    {
        return -1;
    }

    /*
	rLock_Destory(mutex->mutex);
	*/
	OSSemDel(mutex->mutex, 0, 0);
	return VOS_SUCCESS;
}

//static vos_mutex_t critical_section;
void vos_enter_critical_section(void)
{
    //vos_mutex_lock(&critical_section);
    OS_CPU_SR cpu_sr;
    OS_ENTER_CRITICAL();
}

void vos_leave_critical_section(void)
{
    //vos_mutex_unlock(&critical_section);
    OS_CPU_SR cpu_sr;
    OS_EXIT_CRITICAL();
}


vos_uint32_t vos_getpid(void)
{
	
}

static vos_uint32_t generate_thread_id()
{
	static vos_uint32_t thread_id = 100;
	return (thread_id++);
}

vos_status_t vos_thread_create(const char *thread_name,
						vos_thread_proc proc,
						void *arg,
						vos_uint32_t *stack,
						vos_size_t stack_size,
						vos_uint32_t flags,
						vos_int32_t prio,
						vos_thread_t **thread )
{
	vos_int8_t ret;

//    VOS_PRINTF("thread_name[%s], stack_size[%d], prio[%d].\n",
//        thread_name, stack_size, prio);
    
	if(!thread || !proc || !stack)
	{
		return -1;
	}
	
	*thread = VOS_MALLOC_T(vos_thread_t);
	if( NULL == (*thread) )
	{
		return -2;
	}
	
	ret = OSTaskCreate2(proc, arg, stack, prio, stack_size);
	if( OS_NO_ERR != ret)
	{
        VOS_PRINTF("create thread failed, thread_name[%s], ret[%d].\n", thread_name, ret);
		return ret;
	}
#if OS_TASK_NAME_SIZE > 1
	OSTaskNameSet(prio, (INT8U *)thread_name, NULL);
#endif
	
	(*thread)->thread_handle = prio;
	(*thread)->thread_id = generate_thread_id();
	strncpy( (*thread)->obj_name, thread_name, VOS_MAX_OBJ_NAME );

	VOS_PRINTF("thread_name[%s], thread_handle[%d], thread_id[%d], stack_size[%d], prio[%d].\n", 
        (*thread)->obj_name,
        (*thread)->thread_handle,
        (*thread)->thread_id,
        stack_size, prio);
        
	return VOS_SUCCESS;
}

vos_status_t vos_thread_destroy(vos_thread_t *thread)
{
	if(!thread)
	{
		return -1;
	}
	
	OSTaskDel(thread->thread_handle);

	return VOS_SUCCESS;
}

vos_uint32_t vos_msec_to_ticks(vos_uint32_t msecs)
{
	return ((msecs) * (OS_TICKS_PER_SEC)) / 1000;
}

vos_uint32_t vos_ticks_to_msec(vos_uint32_t ticks)
{
	return ((ticks) * 1000) / (OS_TICKS_PER_SEC);
}

vos_status_t vos_thread_sleep(vos_uint32_t msec)
{
	vos_uint32_t ticks = vos_msec_to_ticks(msec);
	OSTimeDly(ticks);
	return VOS_SUCCESS;
}

vos_uint32_t vos_get_local_ip()
{
    struct wlan_network network;
	if (wlan_get_current_network(&network)) 
	{
		return 0;
	}

    return network.address.ip;
}


vos_uint64_t vos_get_system_usec()
{
    return systime_get_us();
}

vos_uint64_t vos_get_system_msec()
{
    unsigned long long us = systime_get_us();
    unsigned long long base = 1000;
    unsigned long long ms = us/base;
    //VOS_PRINTF("vos_get_system_msec-->(%llu, %llu).\n", us, ms);
    return ms;
}

vos_uint64_t vos_get_system_sec()
{
    unsigned long long us = systime_get_us();
    unsigned long long base = 1000000;

    unsigned long long sec = us/base;
    //VOS_PRINTF("vos_get_system_sec-->(%llu, %llu).\n", us, sec);
    
    return sec;
}

vos_int32_t vos_set_system_usec(vos_uint64_t us)
{
    return systime_set_us(us);
}

static vos_uint64_t g_system_start_time = 0;
vos_uint32_t vos_get_system_tick()
{
    
    return (vos_uint32_t)(vos_get_system_msec() - g_system_start_time);
}

vos_uint32_t vos_get_system_tick_sec()
{
    
    return vos_get_system_tick()/1000;
}



/*
#pragma Data(DATA, ".ddccm")
unsigned int dccm_flag = 0x4443434d;  // "DCCM"
#pragma Data()
*/

vos_status_t vos_init(void)
{
/*
    _ASM("sr 0x00,[0x200]"); //set all intterrupt to level 1
	_ASM("sr 0x02,[0x11]");  //enable i-cache
	_ASM("sr 0xc2,[0x48]");  //enable D-CACHE

	soc_init();
	
    // init uart
    Uart_Init(0, BAUDRATE_115200);

    uprintf("FH61 camera version build: %s, %s\n", __DATE__, __TIME__);
    uprintf("ucos start...\n");

    iSys_Init(TAE_BUF_ADDR, TAE_BUF_SIZE);

    // init os
    OSInit();
    
    // init print
    fhprintf_init();


    if(dccm_flag != 0x4443434d)
    {
        uprintf("dccm error!\n");
        dccm_dead();
    }
    
    uprintf("main task runing!\n");

    // app memory pool init
    appmem_init();

    // timer init
    mytimer_init();

    // init sys date-time
    systime_init();
    
    // init and start app
    app_init();
*/
    g_system_start_time = vos_get_system_msec();
    return VOS_SUCCESS;
    
}

void vos_shutdown(void)
{

}


int vos_strcasecmp(const char* s1, const char* s2)
{
    if (!s1) 
    {
        return (s1==s2)?0:1;
    }
    
    if (!s2)
    {
        return 1;
    }
    
	for(; tolower(*s1) == tolower(*s2); ++s1, ++s2)
	{
	    if(*s1 == 0)
	    {
	        return 0;
	    }
	}
	
	return tolower(*(const unsigned char *)s1) - tolower(*(const unsigned char *)s2);
    
}

 char* vos_strdup(const char* str)
{
      vos_size_t len;
      char* copy;

      len = strlen(str) + 1;
      if (!(copy = (char*)VOS_MALLOC_BLK_T(char, len))) return 0;
      memcpy(copy,str,len);
      return copy;
}

vos_uint32_t vos_inet_addr(const char *cp)
{
    return ipaddr_addr(cp);
}

int vos_inet_aton(const char *cp, vos_in_addr *inaddr)
{
    ip_addr_t val;
    
    if(ipaddr_aton(cp, &val))
    {
        inaddr->s_addr = ip4_addr_get_u32(&val);
        VOS_PRINTF("vos_inet_aton---> str_ip=%s, s_addr=0x%x\n", cp, inaddr->s_addr);
        return 1;
    }

    return 0;
}

char* vos_inet_ntoa(vos_in_addr inaddr)
{
    ip_addr_t val;
    
    ip4_addr_set_u32(&val, inaddr.s_addr);

    return ipaddr_ntoa(&val);
}



