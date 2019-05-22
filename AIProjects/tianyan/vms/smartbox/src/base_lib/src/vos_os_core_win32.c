#define __OS_CORE_C__

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
//#include <winnt.h>

#include "vos_types.h"
#include "vos_assert.h"
#include "vos_string.h"
#include "vos_os.h"
#include "vos_time.h"
#include "vos_addr_resolv.h"
#include "vos_log.h"


#if defined(VOS_HAS_WINSOCK_H) && VOS_HAS_WINSOCK_H != 0
#  include <winsock.h>
#endif

#if defined(VOS_HAS_WINSOCK2_H) && VOS_HAS_WINSOCK2_H != 0
#  include <winsock2.h>
#pragma comment(lib,"ws2_32.lib")
#endif

#if defined(VOS_DEBUG_MUTEX) && VOS_DEBUG_MUTEX
#   undef VOS_DEBUG
#   define VOS_DEBUG	    1
#   define LOG_MUTEX(expr)  VOS_LOG(5,expr)
#else
#   define LOG_MUTEX(expr)  VOS_LOG(6,expr)
#endif

#define THIS_FILE	"os_core_win32.c"

#if 1

struct vos_thread_t
{
    char							obj_name[VOS_MAX_OBJ_NAME];
    HANDLE							hthread;
    vos_thread_id_t                 idthread;
    vos_thread_proc					proc;
    void							*arg;
};

struct vos_mutex_t
{
#if VOS_WIN32_WINNT >= 0x0400
    CRITICAL_SECTION				crit;
#else
    HANDLE							hMutex;
#endif
    char							obj_name[VOS_MAX_OBJ_NAME];
};

struct vos_sem_t
{
    HANDLE							hSemaphore;
    char							obj_name[VOS_MAX_OBJ_NAME];
};

struct vos_event_t
{
    HANDLE							hEvent;
    char							obj_name[VOS_MAX_OBJ_NAME];
};

struct vos_atomic_t
{
    long							value;
};

static int							initialized = 0;
//static vos_thread_desc				main_thread;
//static long							thread_tls_id = -1;
static vos_mutex_t					critical_section_mutex;
static unsigned						atexit_count;
static void							(*atexit_func[32])(void);
static vos_status_t					init_mutex(vos_mutex_t *mutex, const char *name);


vos_status_t vos_init(void)
{
    WSADATA							wsa;
    vos_status_t					rc;

	if (initialized) { initialized++;  return VOS_SUCCESS; }

    g_system_start_time = vos_get_system_usec();

    if (WSAStartup(MAKEWORD(2,0), &wsa) != 0) 
    {
	    return VOS_RETURN_OS_ERROR(WSAGetLastError());
    }

	//if (rc=vos_mem_init() != VOS_SUCCESS)
	//{
	//	return rc;
	//}

    /* 初始化线程TLS. */
    //if ((rc=vos_thread_init()) != VOS_SUCCESS)
    //{
	   // return rc;
    //}
    
    if ((rc=vos_log_init()) != VOS_SUCCESS)
    {
        return rc;
    }

    /* 初始化 Critical section. */
	if ((rc = init_mutex(&critical_section_mutex, "pj%p")) != VOS_SUCCESS)
	{
		return rc;
	}

    /* Flag VOS as initialized */
    ++initialized;
    vos_assert(initialized == 1);

    //VOS_LOG(4,(THIS_FILE, "VOS %s for win32 initialized", VOS_VERSION));

    return VOS_SUCCESS;
}

/*
 * vos_atexit()
 */
vos_status_t vos_atexit(void (*func)(void))
{
    if (atexit_count >= VOS_ARRAY_SIZE(atexit_func))
	return -1;

    atexit_func[atexit_count++] = func;
    return VOS_SUCCESS;
}


/*
 * vos_shutdown(void)
 */
void vos_shutdown()
{
    int i;

    vos_assert(initialized > 0);
    if (--initialized != 0)
    {
	    return;
    }

    vos_log_shutdown();

	for (i = atexit_count - 1; i >= 0; --i)
	{
		(*atexit_func[i])();
	}
    atexit_count = 0;

    vos_mutex_destroy(&critical_section_mutex);

    WSACleanup();

	//vos_mem_destroy();	/* release vos pool memory */
}

vos_uint32_t vos_get_local_ip()
{
    vos_sockaddr addr;
    vos_gethostip(AF_INET, &addr);
    return (unsigned int)(((struct sockaddr_in *)&addr)->sin_addr.S_un.S_addr);
}

vos_uint32_t vos_getpid(void)
{
    return GetCurrentProcessId();
}

vos_thread_id_t vos_get_cur_thread_id()
{
    return GetCurrentThreadId();
}

int vos_thread_get_prio(vos_thread_t *thread)
{
    return GetThreadPriority(thread->hthread);
}

vos_status_t vos_thread_set_prio(vos_thread_t *thread,  int prio)
{
    VOS_ASSERT_RETURN(thread, VOS_EINVAL);
    VOS_ASSERT_RETURN(prio>=THREAD_PRIORITY_IDLE && prio<=THREAD_PRIORITY_TIME_CRITICAL,VOS_EINVAL);

    if (SetThreadPriority(thread->hthread, prio) == FALSE)
    {
	    return VOS_RETURN_OS_ERROR(GetLastError());
    }
    
    return VOS_SUCCESS;
}

int vos_thread_get_prio_min(vos_thread_t *thread)
{
    VOS_UNUSED_ARG(thread);
    return THREAD_PRIORITY_IDLE;
}

int vos_thread_get_prio_max(vos_thread_t *thread)
{
    VOS_UNUSED_ARG(thread);
    return THREAD_PRIORITY_TIME_CRITICAL;
}

void* vos_thread_get_os_handle(vos_thread_t *thread) 
{
    VOS_ASSERT_RETURN(thread, NULL);

    return thread->hthread;
}

static DWORD WINAPI thread_main(void* param)
{
    vos_thread_t *rec = (vos_thread_t *)param;
    DWORD result;

    VOS_LOG(6,(rec->obj_name, "Thread started"));

    result = (*rec->proc)(rec->arg);

    VOS_LOG(6,(rec->obj_name, "Thread quitting"));

    return result;
}

/*
 * vos_thread_create(...)
 */
vos_status_t vos_thread_create(const char				*thread_name,
							   vos_thread_proc			proc,
							   void						*arg,
							   vos_uint32_t				*stack,
							   vos_size_t				stack_size,
							   vos_uint32_t				flags,
							   vos_int32_t				prio,
							   vos_thread_t				**thread_ptr)
{
    DWORD dwflags = 0;
    vos_thread_t *rec;

    VOS_ASSERT_RETURN(proc && thread_ptr, VOS_EINVAL);

    if (flags & VOS_THREAD_SUSPENDED)
    {
	    dwflags |= CREATE_SUSPENDED;
    }

    rec = VOS_CALLOC_T(1, vos_thread_t);
    if (!rec)
    {
	    return VOS_ENOMEM;
    }

    if (!thread_name)
    {
	    thread_name = "thr%p";
    }

    if (strchr(thread_name, '%')) 
    {
	    vos_ansi_snprintf(rec->obj_name, VOS_MAX_OBJ_NAME, thread_name, rec);
    } 
    else 
    {
	    vos_ansi_strncpy(rec->obj_name, thread_name, VOS_MAX_OBJ_NAME);
	    rec->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    VOS_LOG(6, (rec->obj_name, "Thread created"));

    rec->proc = proc;
    rec->arg = arg;

	rec->hthread = CreateThread(NULL,
								stack_size,
								thread_main,
								rec,
								dwflags,
								&rec->idthread);
    if (rec->hthread == NULL)
    {
	    return VOS_RETURN_OS_ERROR(GetLastError());
    }

    *thread_ptr = rec;
    return VOS_SUCCESS;
}

const char* vos_thread_get_name(vos_thread_t *p)
{
    vos_thread_t *rec = (vos_thread_t*)p;

    VOS_ASSERT_RETURN(p, "");

    return rec->obj_name;
}

vos_status_t vos_thread_resume(vos_thread_t *p)
{
    vos_thread_t *rec = (vos_thread_t*)p;

    VOS_ASSERT_RETURN(p, VOS_EINVAL);

    if (ResumeThread(rec->hthread) == (DWORD)-1)
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
    else
    {
        return VOS_SUCCESS;
    }
}

vos_status_t vos_thread_join(vos_thread_t *p)
{
    vos_thread_t *rec = (vos_thread_t *)p;
    DWORD rc;

    VOS_ASSERT_RETURN(p, VOS_EINVAL);

    if (p->idthread == vos_get_cur_thread_id())
    {
	    return VOS_ECANCELLED;
    }
    //VOS_LOG(6, (vos_get_cur_thread_id(), "Joining thread %s", p->obj_name));

    rc = WaitForSingleObject(rec->hthread, INFINITE);

    if (rc==WAIT_OBJECT_0)
    {
        return VOS_SUCCESS;
    }
    else if (rc==WAIT_TIMEOUT)
    {
        return VOS_ETIMEDOUT;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_thread_destroy(vos_thread_t *p)
{
    vos_thread_t *rec = (vos_thread_t *)p;

    VOS_ASSERT_RETURN(p, VOS_EINVAL);

    if (CloseHandle(rec->hthread) == TRUE)
    {
        return VOS_SUCCESS;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_thread_sleep(unsigned msec)
{
    Sleep(msec);
    return VOS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
vos_status_t vos_atomic_create( vos_atomic_value_t initial, vos_atomic_t **atomic_ptr)
{
    vos_atomic_t *atomic_var = VOS_MALLOC_T(vos_atomic_t);
    if (!atomic_var)
    {
	    return VOS_ENOMEM;
    }
    
    atomic_var->value = initial;
    *atomic_ptr = atomic_var;

    return VOS_SUCCESS;
}

vos_status_t vos_atomic_destroy( vos_atomic_t *var )
{
    VOS_UNUSED_ARG(var);
    VOS_ASSERT_RETURN(var, VOS_EINVAL);

    return 0;
}

void vos_atomic_set( vos_atomic_t *atomic_var, vos_atomic_value_t value)
{
    InterlockedExchange(&atomic_var->value, value);
}

vos_atomic_value_t vos_atomic_get(vos_atomic_t *atomic_var)
{
    VOS_ASSERT_RETURN(atomic_var, 0);

    return atomic_var->value;
}

vos_atomic_value_t vos_atomic_inc_and_get(vos_atomic_t *atomic_var)
{
#if defined(VOS_WIN32_WINNT) && VOS_WIN32_WINNT >= 0x0400
    return InterlockedIncrement(&atomic_var->value);
#else
    return InterlockedIncrement(&atomic_var->value);
#endif
}

void vos_atomic_inc(vos_atomic_t *atomic_var)
{
    vos_atomic_inc_and_get(atomic_var);
}

vos_atomic_value_t vos_atomic_dec_and_get(vos_atomic_t *atomic_var)
{
#if defined(VOS_WIN32_WINNT) && VOS_WIN32_WINNT >= 0x0400
    return InterlockedDecrement(&atomic_var->value);
#else
    return InterlockedDecrement(&atomic_var->value);
#endif
}

void vos_atomic_dec(vos_atomic_t *atomic_var)
{
    vos_atomic_dec_and_get(atomic_var);
}

void vos_atomic_add( vos_atomic_t *atomic_var, vos_atomic_value_t value )
{
#if defined(VOS_WIN32_WINNT) && VOS_WIN32_WINNT >= 0x0400
    InterlockedExchangeAdd( &atomic_var->value, value );
#else
    InterlockedExchangeAdd( &atomic_var->value, value );
#endif
}

vos_atomic_value_t vos_atomic_add_and_get( vos_atomic_t *atomic_var, vos_atomic_value_t value)
{
#if defined(VOS_WIN32_WINNT) && VOS_WIN32_WINNT >= 0x0400
    long oldValue = InterlockedExchangeAdd( &atomic_var->value, value);
    return oldValue + value;
#else
    long oldValue = InterlockedExchangeAdd( &atomic_var->value, value);
    return oldValue + value;
#endif
}

///////////////////////////////////////////////////////////////////////////////

vos_status_t vos_thread_local_alloc(long *index)
{
    VOS_ASSERT_RETURN(index != NULL, VOS_EINVAL);

    *index = TlsAlloc();

    if (*index == TLS_OUT_OF_INDEXES)
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
    else
    {
        return VOS_SUCCESS;
    }
}

void vos_thread_local_free(long index)
{
    TlsFree(index);
}

vos_status_t vos_thread_local_set(long index, void *value)
{
    BOOL rc;
    rc = TlsSetValue(index, value);
    return rc!=0 ? VOS_SUCCESS : VOS_RETURN_OS_ERROR(GetLastError());
}


void* vos_thread_local_get(long index)
{
    return (void *)TlsGetValue(index);
}

///////////////////////////////////////////////////////////////////////////////
static vos_status_t init_mutex(vos_mutex_t *mutex, const char *name)
{
#if VOS_WIN32_WINNT >= 0x0400
    InitializeCriticalSection(&mutex->crit);
#else
    mutex->hMutex = CreateMutex(NULL, FALSE, NULL);
    if (!mutex->hMutex) {
	return VOS_RETURN_OS_ERROR(GetLastError());
    }
#endif

    /* Set name. */
    if (!name) 
    {
	    name = "mtx%p";
    }
    
    if (strchr(name, '%')) 
    {
	    vos_ansi_snprintf(mutex->obj_name, VOS_MAX_OBJ_NAME, name, mutex);
    } 
    else 
    {
    	vos_ansi_strncpy(mutex->obj_name, name, VOS_MAX_OBJ_NAME);
    	mutex->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    VOS_LOG(6, (mutex->obj_name, "Mutex created"));
    return VOS_SUCCESS;
}

vos_status_t vos_mutex_create(const char *name, int type, vos_mutex_t **mutex_ptr)
{
	vos_status_t rc;
	vos_mutex_t *mutex;

	VOS_UNUSED_ARG(type);
	VOS_ASSERT_RETURN(mutex_ptr, VOS_EINVAL);

	mutex = VOS_MALLOC_T(vos_mutex_t);
	if (!mutex)
	{
		return VOS_ENOMEM;
	}

	rc = init_mutex(mutex, name);
	if (rc != VOS_SUCCESS)
	{
		return rc;
	}

	*mutex_ptr = mutex;

	return VOS_SUCCESS;
}

vos_status_t vos_mutex_create_simple(const char *name, vos_mutex_t **mutex )
{
    return vos_mutex_create(name, VOS_MUTEX_SIMPLE, mutex);
}

vos_status_t vos_mutex_create_recursive(const char *name, vos_mutex_t **mutex )
{
    return vos_mutex_create(name, VOS_MUTEX_RECURSE, mutex);
}

vos_status_t vos_mutex_lock(vos_mutex_t *mutex)
{
    vos_status_t status;

    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

    //LOG_MUTEX((mutex->obj_name, "Mutex: thread %s is waiting", vos_get_cur_thread_id()));

#if VOS_WIN32_WINNT >= 0x0400
    EnterCriticalSection(&mutex->crit);
    status=VOS_SUCCESS;
#else
    if (WaitForSingleObject(mutex->hMutex, INFINITE)==WAIT_OBJECT_0)
    {
        status = VOS_SUCCESS;
    }
    else
    {
        status = VOS_STATUS_FROM_OS(GetLastError());
    }
#endif
    //LOG_MUTEX((mutex->obj_name, (status==VOS_SUCCESS ? "Mutex acquired by thread %s" : "FAILED by %s"), vos_get_cur_thread_id()));

    return status;
}

vos_status_t vos_mutex_unlock(vos_mutex_t *mutex)
{
    vos_status_t status;

    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

    LOG_MUTEX((mutex->obj_name, "Mutex released by thread %u", vos_get_cur_thread_id()));

#if VOS_WIN32_WINNT >= 0x0400
    LeaveCriticalSection(&mutex->crit);
    status=VOS_SUCCESS;
#else
    status = ReleaseMutex(mutex->hMutex) ? VOS_SUCCESS : 
                VOS_STATUS_FROM_OS(GetLastError());
#endif
    return status;
}

vos_status_t vos_mutex_trylock(vos_mutex_t *mutex)
{
    vos_status_t status;

    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

    LOG_MUTEX((mutex->obj_name, "Mutex: thread %u is trying", vos_get_cur_thread_id()));

#if VOS_WIN32_WINNT >= 0x0400
    status=TryEnterCriticalSection(&mutex->crit) ? VOS_SUCCESS : -1;
#else
    status = WaitForSingleObject(mutex->hMutex, 0)==WAIT_OBJECT_0 ? 
                VOS_SUCCESS : VOS_ETIMEDOUT;
#endif
    if (status==VOS_SUCCESS) 
    {
    	LOG_MUTEX((mutex->obj_name, "Mutex acquired by thread %u", vos_get_cur_thread_id()));

    #if VOS_DEBUG
    	mutex->owner = vos_thread_this();
    	++mutex->nesting_level;
    #endif
    } 
    else 
    {
    	LOG_MUTEX((mutex->obj_name, "Mutex: thread %u's trylock() failed", vos_get_cur_thread_id()));
    }

    return status;
}

vos_status_t vos_mutex_destroy(vos_mutex_t *mutex)
{
	vos_status_t status = VOS_SUCCESS;

    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

    LOG_MUTEX((mutex->obj_name, "Mutex destroyed"));

#if VOS_WIN32_WINNT >= 0x0400
    DeleteCriticalSection(&mutex->crit);
    return status;
#else

	status = CloseHandle(mutex->hMutex) ? VOS_SUCCESS : VOS_RETURN_OS_ERROR(GetLastError());
	VOS_FREE_T(mutex);
	return status;
#endif
}

void vos_enter_critical_section(void)
{
    vos_mutex_lock(&critical_section_mutex);
}

void vos_leave_critical_section(void)
{
    vos_mutex_unlock(&critical_section_mutex);
}

#if defined(VOS_HAS_SEMAPHORE) && VOS_HAS_SEMAPHORE != 0
vos_status_t vos_sem_create(const char *name,
							unsigned initial,
							unsigned max,
							vos_sem_t **sem_ptr)
{
    vos_sem_t *sem;

    VOS_ASSERT_RETURN(sem_ptr, VOS_EINVAL);

    sem = VOS_MALLOC_T(vos_sem_t);
    sem->hSemaphore = CreateSemaphore(NULL, initial, max, NULL);
    if (!sem->hSemaphore)
    {
	    return VOS_RETURN_OS_ERROR(GetLastError());
    }
    
    if (!name) 
    {
	    name = "sem%p";
    }
    
    if (strchr(name, '%')) 
    {
	    vos_ansi_snprintf(sem->obj_name, VOS_MAX_OBJ_NAME, name, sem);
    } 
    else 
    {
	    vos_ansi_strncpy(sem->obj_name, name, VOS_MAX_OBJ_NAME);
	    sem->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    LOG_MUTEX((sem->obj_name, "Semaphore created"));

    *sem_ptr = sem;
    return VOS_SUCCESS;
}

static vos_status_t vos_sem_wait_for(vos_sem_t *sem, unsigned timeout)
{
    DWORD result;

    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    LOG_MUTEX((sem->obj_name, "Semaphore: thread %u is waiting", vos_get_cur_thread_id()));

    result = WaitForSingleObject(sem->hSemaphore, timeout);
    if (result == WAIT_OBJECT_0) 
    {
    	LOG_MUTEX((sem->obj_name, "Semaphore acquired by thread %u", vos_get_cur_thread_id()));
    } 
    else 
    {
    	LOG_MUTEX((sem->obj_name, "Semaphore: thread %u FAILED to acquire", vos_get_cur_thread_id()));
    }

	if (result == WAIT_OBJECT_0)
	{
		return VOS_SUCCESS;
	}
	else if (result == WAIT_TIMEOUT)
	{
		return VOS_ETIMEDOUT;
	}
	else
	{
		return VOS_RETURN_OS_ERROR(GetLastError());
	}
}

vos_status_t vos_sem_wait(vos_sem_t *sem)
{
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    return vos_sem_wait_for(sem, INFINITE);
}

vos_status_t vos_sem_trywait(vos_sem_t *sem)
{
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    return vos_sem_wait_for(sem, 0);
}

vos_status_t vos_sem_post(vos_sem_t *sem)
{
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    LOG_MUTEX((sem->obj_name, "Semaphore released by thread %u", vos_get_cur_thread_id()));

    if (ReleaseSemaphore(sem->hSemaphore, 1, NULL))
    {
        return VOS_SUCCESS;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_sem_destroy(vos_sem_t *sem)
{
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    LOG_MUTEX((sem->obj_name, "Semaphore destroyed by thread %u", vos_get_cur_thread_id()));

    if (CloseHandle(sem->hSemaphore))
    {
        return VOS_SUCCESS;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

#endif	/* VOS_HAS_SEMAPHORE */
///////////////////////////////////////////////////////////////////////////////


#if defined(VOS_HAS_EVENT_OBJ) && VOS_HAS_EVENT_OBJ != 0
vos_status_t vos_event_create(const char *name,
							  vos_bool_t manual_reset,
							  vos_bool_t initial,
							  vos_event_t **event_ptr)
{
    vos_event_t *event;

    VOS_ASSERT_RETURN(event_ptr, VOS_EINVAL);

    event = VOS_MALLOC_T(vos_event_t);
    if (!event)
    {
        return VOS_ENOMEM;
    }
    
    event->hEvent = CreateEvent(NULL, manual_reset?TRUE:FALSE, initial?TRUE:FALSE, NULL);

    if (!event->hEvent)
    {
	    return VOS_RETURN_OS_ERROR(GetLastError());
    }
    
    if (!name) 
    {
	    name = "evt%p";
    }
    
    if (strchr(name, '%')) 
    {
	    vos_ansi_snprintf(event->obj_name, VOS_MAX_OBJ_NAME, name, event);
    } 
    else 
    {
    	vos_ansi_strncpy(event->obj_name, name, VOS_MAX_OBJ_NAME);
    	event->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    VOS_LOG(6, (event->obj_name, "Event created"));

    *event_ptr = event;
    return VOS_SUCCESS;
}

static vos_status_t vos_event_wait_for(vos_event_t *event, unsigned timeout)
{
    DWORD result;

    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    VOS_LOG(6, (event->obj_name, "Event: thread %u is waiting", vos_get_cur_thread_id()));

    result = WaitForSingleObject(event->hEvent, timeout);
    if (result == WAIT_OBJECT_0) 
    {
    	VOS_LOG(6, (event->obj_name, "Event: thread %u is released", vos_get_cur_thread_id()));
    }
    else
    {
    	VOS_LOG(6, (event->obj_name, "Event: thread %u FAILED to acquire", vos_get_cur_thread_id()));
    }

    if (result == WAIT_OBJECT_0)
    {
        return VOS_SUCCESS;
    }
    else if (result == WAIT_TIMEOUT)
    {
        return VOS_ETIMEDOUT;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_event_wait(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    return vos_event_wait_for(event, INFINITE);
}

vos_status_t vos_event_trywait(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    return vos_event_wait_for(event, 0);
}

vos_status_t vos_event_set(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    VOS_LOG(6, (event->obj_name, "Setting event"));

    if (SetEvent(event->hEvent))
    {
        return VOS_SUCCESS;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_event_pulse(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    VOS_LOG(6, (event->obj_name, "Pulsing event"));

    if (PulseEvent(event->hEvent))
    {
        return VOS_SUCCESS;
    }
    else
    {
        return VOS_RETURN_OS_ERROR(GetLastError());
    }
}

vos_status_t vos_event_reset(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    VOS_LOG(6, (event->obj_name, "Event is reset"));

    if (ResetEvent(event->hEvent))
    {
        return VOS_SUCCESS;
    }

    return VOS_RETURN_OS_ERROR(GetLastError());
}

vos_status_t vos_event_destroy(vos_event_t *event)
{
    VOS_ASSERT_RETURN(event, VOS_EINVAL);

    VOS_LOG(6, (event->obj_name, "Event is destroying"));

    if (CloseHandle(event->hEvent))
    {
        return VOS_SUCCESS;
    }
    
    return VOS_RETURN_OS_ERROR(GetLastError());
}

#endif	/* VOS_HAS_EVENT_OBJ */

int vos_run_app(vos_main_func_ptr main_func, int argc, char *argv[], unsigned flags)
{
    VOS_UNUSED_ARG(flags);
    return (*main_func)(argc, argv);
}

#endif