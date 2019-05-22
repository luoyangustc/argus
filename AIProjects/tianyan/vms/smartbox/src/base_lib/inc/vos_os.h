
#ifndef __VOS_OS_H__
#define __VOS_OS_H__

#include "vos_types.h"

#undef	EXT
#ifndef __OS_CORE_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

#if (OS_WIN32 == 1)
    #define vos_get_native_netos_error()	                    WSAGetLastError()
#elif(OS_LINUX == 1)
    #define vos_get_native_netos_error()	                    errno
#else
    #define vos_get_native_netos_error()
#endif

#ifndef OS_UCOS_II
    #define appmem_dump()
    typedef unsigned int OS_STK;
#endif

#if (OS_WIN32 == 1)
typedef DWORD vos_thread_id_t;
#else
typedef unsigned int vos_thread_id_t;
#endif

typedef int   vos_thread_ret_t;
typedef void*   vos_thread_arg_t;
typedef vos_thread_ret_t    (*vos_thread_proc)(vos_thread_arg_t arg);

#if !defined(VOS_THREAD_DESC_SIZE)
	#define VOS_THREAD_DESC_SIZE								(64)
#endif
typedef long													vos_thread_desc[VOS_THREAD_DESC_SIZE];

enum vos_thread_create_flags
{
	VOS_THREAD_SUSPENDED = 1
};

typedef enum vos_mutex_type_e
{
	VOS_MUTEX_DEFAULT,
	VOS_MUTEX_SIMPLE,
	VOS_MUTEX_RECURSE
} vos_mutex_type_e;

EXT vos_status_t vos_init(void);
EXT void vos_shutdown(void);

typedef int(*vos_main_func_ptr)(int argc, char *argv[]);
EXT int vos_run_app(vos_main_func_ptr main_func, int argc, char *argv[], unsigned flags);

//线程相关
EXT vos_status_t vos_thread_init(void);
vos_status_t vos_thread_create(     const char				*thread_name,
                                    vos_thread_proc			proc,
                                    void					*arg,
                                    vos_uint32_t			*stack,
                                    vos_size_t				stack_size,
                                    vos_uint32_t			flags,
                                    vos_int32_t				prio,
                                    vos_thread_t			**thread_ptr);

EXT vos_status_t vos_thread_destroy(vos_thread_t *thread);
EXT vos_status_t vos_thread_sleep(unsigned msec);
EXT vos_status_t vos_thread_join(vos_thread_t *p);
EXT int vos_thread_get_prio(vos_thread_t *thread);
EXT vos_status_t vos_thread_set_prio(vos_thread_t *thread,  int prio);
EXT void* vos_thread_get_os_handle(vos_thread_t *thread);
EXT const char* vos_thread_get_name(vos_thread_t *p);

//TLS操作
EXT vos_status_t vos_thread_local_alloc(long *index);
EXT void* vos_thread_local_get(long index);
EXT vos_status_t vos_thread_local_set(long index, void *value);
EXT void vos_thread_local_free(long index);

//原子操作
EXT vos_status_t vos_atomic_create( vos_atomic_value_t initial, vos_atomic_t **atomic_ptr);
EXT void vos_atomic_set( vos_atomic_t *atomic_var, vos_atomic_value_t value);
EXT vos_atomic_value_t vos_atomic_get(vos_atomic_t *atomic_var);
EXT vos_atomic_value_t vos_atomic_inc_and_get(vos_atomic_t *atomic_var);
EXT void vos_atomic_inc(vos_atomic_t *atomic_var);
EXT vos_atomic_value_t vos_atomic_dec_and_get(vos_atomic_t *atomic_var);
EXT void vos_atomic_dec(vos_atomic_t *atomic_var);
EXT void vos_atomic_add( vos_atomic_t *atomic_var, vos_atomic_value_t value );
EXT vos_atomic_value_t vos_atomic_add_and_get( vos_atomic_t *atomic_var, vos_atomic_value_t value);
EXT vos_status_t vos_atomic_destroy( vos_atomic_t *var );

//关键区操作
EXT void vos_enter_critical_section(void);
EXT void vos_leave_critical_section(void);

//互斥量操作
EXT vos_status_t vos_mutex_create(const char *name, int type, vos_mutex_t **mutex);
EXT vos_status_t vos_mutex_create_simple(const char *name, vos_mutex_t **mutex );
EXT vos_status_t vos_mutex_create_recursive(  const char *name, vos_mutex_t **mutex );
EXT vos_status_t vos_mutex_lock(vos_mutex_t *mutex);
EXT vos_status_t vos_mutex_trylock(vos_mutex_t *mutex);
EXT vos_status_t vos_mutex_unlock(vos_mutex_t *mutex);
EXT vos_status_t vos_mutex_destroy(vos_mutex_t *mutex);

//信号量操作
#if defined(VOS_HAS_SEMAPHORE) && VOS_HAS_SEMAPHORE != 0
EXT vos_status_t vos_sem_create(const char *name, unsigned initial, unsigned max, vos_sem_t **sem_ptr);
EXT vos_status_t vos_sem_wait(vos_sem_t *sem);
EXT vos_status_t vos_sem_trywait(vos_sem_t *sem);
EXT vos_status_t vos_sem_post(vos_sem_t *sem);
EXT vos_status_t vos_sem_destroy(vos_sem_t *sem);
#endif // defined(VOS_HAS_SEMAPHORE) && VOS_HAS_SEMAPHORE != 0

//事件对象操作
#if defined(VOS_HAS_EVENT_OBJ) && VOS_HAS_EVENT_OBJ != 0
EXT vos_status_t vos_event_create(const char *name, vos_bool_t manual_reset, vos_bool_t initial, vos_event_t **event_ptr);
EXT vos_status_t vos_event_wait(vos_event_t *event);
EXT vos_status_t vos_event_trywait(vos_event_t *event);
EXT vos_status_t vos_event_set(vos_event_t *event);
EXT vos_status_t vos_event_pulse(vos_event_t *event);
EXT vos_status_t vos_event_reset(vos_event_t *event);
EXT vos_status_t vos_event_destroy(vos_event_t *event);
#endif	/* VOS_HAS_EVENT_OBJ */

#if defined(VOS_HAS_HIGH_RES_TIMER) && VOS_HAS_HIGH_RES_TIMER != 0
vos_status_t vos_get_timestamp(vos_timestamp_t *ts);
#endif

//其他
EXT vos_uint32_t vos_getpid(void);
EXT vos_thread_id_t vos_get_cur_thread_id();
EXT vos_uint32_t vos_get_local_ip();

EXT vos_uint64_t g_system_start_time;

VOS_END_DECL

#endif

