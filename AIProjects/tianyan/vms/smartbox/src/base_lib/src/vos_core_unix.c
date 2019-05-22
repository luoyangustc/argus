#include "vos_config.h"
#if (OS_LINUX == 1)
#define __OS_CORE_C__

#include "vos_os.h"
#include "vos_assert.h"
#include "vos_log.h"
#include "vos_string.h"
#include "vos_guid.h"
#include "vos_time.h"
#include "vos_addr_resolv.h"
#include "vos_socket.h"

#include <unistd.h>	    // getpid()
#include <errno.h>	    // errno

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <signal.h>

#define THIS_FILE   "os_core_unix.c"

#define SIGNATURE1  0xDEAFBEEF
#define SIGNATURE2  0xDEADC0DE

struct vos_thread_t
{
    char	    obj_name[VOS_MAX_OBJ_NAME];
    pthread_t	    thread;
    vos_thread_proc proc;
    void	   *arg;
    vos_uint32_t	    signature1;
    vos_uint32_t	    signature2;

    vos_mutex_t	   *suspended_mutex;

#if defined(VOS_OS_HAS_CHECK_STACK) && VOS_OS_HAS_CHECK_STACK!=0
    vos_uint32_t	    stk_size;
    vos_uint32_t	    stk_max_usage;
    char	   *stk_start;
    const char	   *caller_file;
    int		    caller_line;
#endif
};

struct vos_atomic_t
{
    vos_mutex_t	       *mutex;
    vos_atomic_value_t	value;
};

struct vos_mutex_t
{
    pthread_mutex_t     mutex;
    char		obj_name[VOS_MAX_OBJ_NAME];
#if VOS_DEBUG
    int		        nesting_level;
    vos_thread_t	       *owner;
    char		owner_name[VOS_MAX_OBJ_NAME];
#endif
};

#if defined(VOS_HAS_SEMAPHORE) && VOS_HAS_SEMAPHORE != 0
struct vos_sem_t
{
    sem_t	       *sem;
    char		obj_name[VOS_MAX_OBJ_NAME];
};
#endif /* VOS_HAS_SEMAPHORE */

#if defined(VOS_HAS_EVENT_OBJ) && VOS_HAS_EVENT_OBJ != 0
struct vos_event_t
{
    char		obj_name[VOS_MAX_OBJ_NAME];
};
#endif	/* VOS_HAS_EVENT_OBJ */


/*
 * Flag and reference counter for VOS instance.
 */
static int initialized;

#if VOS_HAS_THREADS
    static vos_thread_t main_thread;
    static long thread_tls_id;
    static vos_mutex_t critical_section;
#else
#   define MAX_THREADS 32
    static int tls_flag[MAX_THREADS];
    static void *tls[MAX_THREADS];
#endif

static unsigned atexit_count;
static void (*atexit_func[32])(void);

static vos_status_t init_mutex(vos_mutex_t *mutex, const char *name, int type);

#define USEC_PER_SEC	1000000

vos_status_t vos_get_timestamp(vos_timestamp_t *ts)
{
    struct timeval tv;

    if (gettimeofday(&tv, NULL) != 0) {
        return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
    }

    ts->u64 = tv.tv_sec;
    ts->u64 *= USEC_PER_SEC;
    ts->u64 += tv.tv_usec;
    return VOS_SUCCESS;
}

/*
 * Init VOS!
 */
vos_status_t vos_init(void)
{
    char dummy_guid[VOS_GUID_MAX_LENGTH];
    vos_str_t guid;
    vos_status_t rc;

    /* Check if VOS have been initialized */
    if (initialized) 
    {
    	++initialized;
    	return VOS_SUCCESS;
    }

#if VOS_HAS_THREADS
    /* Init this thread's TLS. */
    if ((rc=vos_thread_init()) != 0) 
    {
	    return rc;
    }

    /* Critical section. */
    if ((rc=init_mutex(&critical_section, "critsec", VOS_MUTEX_RECURSE)) != 0)
    {
	    return rc;
    }
    
#endif

    /* Init logging */
    vos_log_init();

    /* Initialize exception ID for the pool. 
     * Must do so after critical section is configured.
     */
    /*rc = vos_exception_id_alloc("VOS/No memory", &VOS_NO_MEMORY_EXCEPTION);
    if (rc != VOS_SUCCESS)
        return rc;*/
    
    /* Init random seed. */
    /* Or probably not. Let application in charge of this */
    /* vos_srand( clock() ); */

    /* Startup GUID. */
    guid.ptr = dummy_guid;
    vos_generate_unique_string( &guid );

    /* Startup timestamp */
#if defined(VOS_HAS_HIGH_RES_TIMER) && VOS_HAS_HIGH_RES_TIMER != 0
    {
		vos_timestamp_t dummy_ts;
		if ((rc=vos_get_timestamp(&dummy_ts)) != 0) 
		{
			return rc;
		}
    }
#endif   

    /* Flag VOS as initialized */
    ++initialized;
    vos_assert(initialized == 1);

    VOS_LOG(4,(THIS_FILE, "VOS %s for POSIX initialized",
	      VOS_VERSION));

    return VOS_SUCCESS;
}

/*
 * vos_atexit()
 */
vos_status_t vos_atexit(void (*func)(void))
{
    if (atexit_count >= VOS_ARRAY_SIZE(atexit_func))
	return VOS_ETOOMANY;

    atexit_func[atexit_count++] = func;
    return VOS_SUCCESS;
}

/*
 * vos_shutdown(void)
 */
void vos_shutdown()
{
    int i;

    /* Only perform shutdown operation when 'initialized' reaches zero */
    vos_assert(initialized > 0);
    if (--initialized != 0)
	return;

    /* Call atexit() functions */
    for (i=atexit_count-1; i>=0; --i) {
	(*atexit_func[i])();
    }
    atexit_count = 0;

    /* Free exception ID */
   /* if (VOS_NO_MEMORY_EXCEPTION != -1) {
	vos_exception_id_free(VOS_NO_MEMORY_EXCEPTION);
	VOS_NO_MEMORY_EXCEPTION = -1;
    }*/

#if VOS_HAS_THREADS
    /* Destroy VOS critical section */
    vos_mutex_destroy(&critical_section);

    /* Free VOS TLS */
    if (thread_tls_id != -1) {
	vos_thread_local_free(thread_tls_id);
	thread_tls_id = -1;
    }

    /* Ticket #1132: Assertion when (re)starting VOS on different thread */
    vos_bzero(&main_thread, sizeof(main_thread));
#endif

    /* Clear static variables */
    //vos_errno_clear_handlers();

	/* release vos pool memory */
	//vos_mem_destroy();
}


/*
 * vos_getpid(void)
 */
vos_uint32_t vos_getpid(void)
{
    ////VOS_CHECK_STACK;
    return getpid();
}

vos_thread_id_t vos_get_cur_thread_id()
{
    return syscall(SYS_gettid);
}

vos_uint32_t vos_get_local_ip()
{
    vos_sockaddr addr;

    vos_gethostip(AF_INET, &addr);

    return (*(unsigned int*)&addr.ipv4.sin_addr);
}

/*
 * Check if this thread has been registered to VOS.
 */
vos_bool_t vos_thread_is_registered(void)
{
#if VOS_HAS_THREADS
    return vos_thread_local_get(thread_tls_id) != 0;
#else
    vos_assert("vos_thread_is_registered() called in non-threading mode!");
    return true;
#endif
}


/*
 * Get thread priority value for the thread.
 */
int vos_thread_get_prio(vos_thread_t *thread)
{
#if VOS_HAS_THREADS
    struct sched_param param;
    int policy;
    int rc;

    rc = pthread_getschedparam (thread->thread, &policy, &param);
    if (rc != 0)
	return -1;

    return param.sched_priority;
#else
    VOS_UNUSED_ARG(thread);
    return 1;
#endif
}


/*
 * Set the thread priority.
 */
vos_status_t vos_thread_set_prio(vos_thread_t *thread,  int prio)
{
#if VOS_HAS_THREADS
    struct sched_param param;
    int policy;
    int rc;

    rc = pthread_getschedparam (thread->thread, &policy, &param);
    if (rc != 0)
	return VOS_RETURN_OS_ERROR(rc);

    param.sched_priority = prio;

    rc = pthread_setschedparam(thread->thread, policy, &param);
    if (rc != 0)
	return VOS_RETURN_OS_ERROR(rc);

    return VOS_SUCCESS;
#else
    VOS_UNUSED_ARG(thread);
    VOS_UNUSED_ARG(prio);
    vos_assert("vos_thread_set_prio() called in non-threading mode!");
    return 1;
#endif
}


/*
 * Get the lowest priority value available on this system.
 */
int vos_thread_get_prio_min(vos_thread_t *thread)
{
    struct sched_param param;
    int policy;
    int rc;

    rc = pthread_getschedparam(thread->thread, &policy, &param);
    if (rc != 0)
	return -1;

#if defined(_POSIX_PRIORITY_SCHEDULING)
    return sched_get_priority_min(policy);
#elif defined __OpenBSD__
    /* Thread prio min/max are declared in OpenBSD private hdr */
    return 0;
#else
    vos_assert("vos_thread_get_prio_min() not supported!");
    return 0;
#endif
}


/*
 * Get the highest priority value available on this system.
 */
int vos_thread_get_prio_max(vos_thread_t *thread)
{
    struct sched_param param;
    int policy;
    int rc;

    rc = pthread_getschedparam(thread->thread, &policy, &param);
    if (rc != 0)
	return -1;

#if defined(_POSIX_PRIORITY_SCHEDULING)
    return sched_get_priority_max(policy);
#elif defined __OpenBSD__
    /* Thread prio min/max are declared in OpenBSD private hdr */
    return 31;
#else
    vos_assert("vos_thread_get_prio_max() not supported!");
    return 0;
#endif
}


/*
 * Get native thread handle
 */
void* vos_thread_get_os_handle(vos_thread_t *thread) 
{
    VOS_ASSERT_RETURN(thread, NULL);

#if VOS_HAS_THREADS
    return &thread->thread;
#else
    vos_assert("vos_thread_is_registered() called in non-threading mode!");
    return NULL;
#endif
}

/*
 * vos_thread_register(..)
 */
vos_status_t vos_thread_register ( const char *cstr_thread_name,
					 vos_thread_desc desc,
					 vos_thread_t **ptr_thread)
{
#if VOS_HAS_THREADS
    char stack_ptr;
    vos_status_t rc;
    vos_thread_t *thread = (vos_thread_t *)desc;
    vos_str_t thread_name = vos_str((char*)cstr_thread_name);

    /* Size sanity check. */
    if (sizeof(vos_thread_desc) < sizeof(vos_thread_t)) {
	vos_assert(!"Not enough vos_thread_desc size!");
	return VOS_EBUG;
    }

    /* Warn if this thread has been registered before */
    if (vos_thread_local_get (thread_tls_id) != 0) {
	// 2006-02-26 bennylp:
	//  This wouldn't work in all cases!.
	//  If thread is created by external module (e.g. sound thread),
	//  thread may be reused while the pool used for the thread descriptor
	//  has been deleted by application.
	//*thread_ptr = (vos_thread_t*)vos_thread_local_get (thread_tls_id);
        //return VOS_SUCCESS;
	VOS_LOG(4,(THIS_FILE, "Info: possibly re-registering existing "
			     "thread"));
    }

    /* On the other hand, also warn if the thread descriptor buffer seem to
     * have been used to register other threads.
     */
    vos_assert(thread->signature1 != SIGNATURE1 ||
	      thread->signature2 != SIGNATURE2 ||
	      (thread->thread == pthread_self()));

    /* Initialize and set the thread entry. */
    vos_bzero(desc, sizeof(struct vos_thread_t));
    thread->thread = pthread_self();
    thread->signature1 = SIGNATURE1;
    thread->signature2 = SIGNATURE2;

    if(cstr_thread_name && thread_name.slen < sizeof(thread->obj_name)-1)
	vos_ansi_snprintf(thread->obj_name, sizeof(thread->obj_name), 
			 cstr_thread_name, thread->thread);
    else
	vos_ansi_snprintf(thread->obj_name, sizeof(thread->obj_name), 
			 "thr%p", (void*)thread->thread);
    
    rc = vos_thread_local_set(thread_tls_id, thread);
    if (rc != VOS_SUCCESS) {
	vos_bzero(desc, sizeof(struct vos_thread_t));
	return rc;
    }

#if defined(VOS_OS_HAS_CHECK_STACK) && VOS_OS_HAS_CHECK_STACK!=0
    thread->stk_start = &stack_ptr;
    thread->stk_size = 0xFFFFFFFFUL;
    thread->stk_max_usage = 0;
#else
    stack_ptr = '\0';
#endif

    *ptr_thread = thread;
    return VOS_SUCCESS;
#else
    vos_thread_t *thread = (vos_thread_t*)desc;
    *ptr_thread = thread;
    return VOS_SUCCESS;
#endif
}

/*
 * vos_thread_init(void)
 */
vos_status_t vos_thread_init(void)
{
#if VOS_HAS_THREADS
    vos_status_t rc;
    vos_thread_t *dummy;

    rc = vos_thread_local_alloc(&thread_tls_id );
    if (rc != VOS_SUCCESS) 
    {
	    return rc;
    }
    return vos_thread_register("thr%p", (long*)&main_thread, &dummy);
#else
    VOS_LOG(2,(THIS_FILE, "Thread init error. Threading is not enabled!"));
    return VOS_EINVALIDOP;
#endif
}

#if VOS_HAS_THREADS
/*
 * thread_main()
 *
 * This is the main entry for all threads.
 */
static void *thread_main(void *param)
{
    vos_thread_t *rec = (vos_thread_t*)param;
    void *result;
    vos_status_t rc;

    sigset_t signal_mask; 
    sigemptyset(&signal_mask); 
    sigaddset(&signal_mask, SIGPIPE); 
    if(pthread_sigmask(SIG_BLOCK, &signal_mask, NULL) == -1)
    {
        VOS_LOG(6,(rec->obj_name, "block sigpipe error\n"));
    }

#if defined(VOS_OS_HAS_CHECK_STACK) && VOS_OS_HAS_CHECK_STACK!=0
    rec->stk_start = (char*)&rec;
#endif

    /* Set current thread id. */
    rc = vos_thread_local_set(thread_tls_id, rec);
    if (rc != VOS_SUCCESS) 
    {
	    vos_assert(!"Thread TLS ID is not set (vos_init() error?)");
    }

    /* Check if suspension is required. */
    if (rec->suspended_mutex) 
    {
    	vos_mutex_lock(rec->suspended_mutex);
    	vos_mutex_unlock(rec->suspended_mutex);
    }

    VOS_LOG(6,(rec->obj_name, "Thread started"));

    /* Call user's entry! */
    result = (*rec->proc)(rec->arg);

    /* Done. */
    VOS_LOG(6,(rec->obj_name, "Thread quitting"));

    return result;
}
#endif

vos_status_t vos_thread_create(const char				*thread_name,
    vos_thread_proc			proc,
    void						*arg,
    vos_uint32_t				*stack,
    vos_size_t				stack_size,
    vos_uint32_t				flags,
    vos_int32_t				prio,
    vos_thread_t				**thread_ptr)
{
#if VOS_HAS_THREADS
    vos_thread_t *rec;
    pthread_attr_t thread_attr;
    void *stack_addr;
    int rc;

    VOS_UNUSED_ARG(stack_addr);

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(proc && thread_ptr, VOS_EINVAL);

    /* Create thread record and assign name for the thread */
    rec = VOS_CALLOC_T(1, struct vos_thread_t);
    VOS_ASSERT_RETURN(rec, VOS_ENOMEM);
    
    /* Set name. */
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
    	strncpy(rec->obj_name, thread_name, VOS_MAX_OBJ_NAME);
    	rec->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    /* Set default stack size */
    if (stack_size == 0)
    {
        stack_size = VOS_THREAD_DEFAULT_STACK_SIZE;
    }
    
#if defined(VOS_OS_HAS_CHECK_STACK) && VOS_OS_HAS_CHECK_STACK!=0
    rec->stk_size = stack_size;
    rec->stk_max_usage = 0;
#endif

    /* Emulate suspended thread with mutex. */
    if (flags & VOS_THREAD_SUSPENDED) 
    {
    	rc = vos_mutex_create_simple(NULL, &rec->suspended_mutex);
    	if (rc != VOS_SUCCESS) 
    	{
    	    return rc;
    	}

    	vos_mutex_lock(rec->suspended_mutex);
    }
    else 
    {
        vos_assert(rec->suspended_mutex == NULL);
    }
    

    /* Init thread attributes */
    pthread_attr_init(&thread_attr);

#if defined(VOS_THREAD_SET_STACK_SIZE) && VOS_THREAD_SET_STACK_SIZE!=0
    /* Set thread's stack size */
    rc = pthread_attr_setstacksize(&thread_attr, stack_size);
    if (rc != 0)
    {
	    return VOS_RETURN_OS_ERROR(rc);
	}
#endif	/* VOS_THREAD_SET_STACK_SIZE */


#if defined(VOS_THREAD_ALLOCATE_STACK) && VOS_THREAD_ALLOCATE_STACK!=0
    /* Allocate memory for the stack */
    stack_addr = malloc(stack_size);
    VOS_ASSERT_RETURN(stack_addr, VOS_ENOMEM);

    rc = pthread_attr_setstackaddr(&thread_attr, stack_addr);
    if (rc != 0)
    {
	    return VOS_RETURN_OS_ERROR(rc);
	}
#endif	/* VOS_THREAD_ALLOCATE_STACK */


    /* Create the thread. */
    rec->proc = proc;
    rec->arg = arg;
    rc = pthread_create( &rec->thread, &thread_attr, &thread_main, rec);
    if (rc != 0) 
    {
	    return VOS_RETURN_OS_ERROR(rc);
    }

    *thread_ptr = rec;

    VOS_LOG(6, (rec->obj_name, "Thread created"));
    return VOS_SUCCESS;
#else
    vos_assert(!"Threading is disabled!");
    return VOS_EINVALIDOP;
#endif
}

/*
 * vos_thread-get_name()
 */
const char* vos_thread_get_name(vos_thread_t *p)
{
#if VOS_HAS_THREADS
    vos_thread_t *rec = (vos_thread_t*)p;

    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(p, "");

    return rec->obj_name;
#else
    return "";
#endif
}

/*
 * vos_thread_resume()
 */
vos_status_t vos_thread_resume(vos_thread_t *p)
{
    vos_status_t rc;

    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(p, VOS_EINVAL);

    rc = vos_mutex_unlock(p->suspended_mutex);

    return rc;
}

/*
 * vos_thread_this()
 */
vos_thread_t* vos_thread_this(void)
{
#if VOS_HAS_THREADS
    vos_thread_t *rec = (vos_thread_t*)vos_thread_local_get(thread_tls_id);
    
    if (rec == NULL) {
	vos_assert(!"Calling VOS from unknown/external thread. You must "
		   "register external threads with vos_thread_register() "
		   "before calling any VOS functions.");
    }

    /*
     * MUST NOT check stack because this function is called
     * by //VOS_CHECK_STACK itself!!!
     *
     */

    return rec;
#else
    vos_assert(!"Threading is not enabled!");
    return NULL;
#endif
}

/*
 * vos_thread_join()
 */
vos_status_t vos_thread_join(vos_thread_t *p)
{
#if VOS_HAS_THREADS
    vos_thread_t *rec = (vos_thread_t *)p;
    void *ret;
    int result;

    //VOS_CHECK_STACK;

    VOS_LOG(6, (vos_thread_this()->obj_name, "Joining thread %s", p->obj_name));
    result = pthread_join( rec->thread, &ret);

    if (result == 0)
	return VOS_SUCCESS;
    else {
	/* Calling pthread_join() on a thread that no longer exists and 
	 * getting back ESRCH isn't an error (in this context). 
	 * Thanks Phil Torre <ptorre@zetron.com>.
	 */
	return result==ESRCH ? VOS_SUCCESS : VOS_RETURN_OS_ERROR(result);
    }
#else
    //VOS_CHECK_STACK;
    vos_assert(!"No multithreading support!");
    return VOS_EINVALIDOP;
#endif
}

/*
 * vos_thread_destroy()
 */
vos_status_t vos_thread_destroy(vos_thread_t *p)
{
    //VOS_CHECK_STACK;

    /* Destroy mutex used to suspend thread */
    if (p->suspended_mutex) 
    {
    	vos_mutex_destroy(p->suspended_mutex);
    	p->suspended_mutex = NULL;
    }

    return VOS_SUCCESS;
}

/*
 * vos_thread_sleep()
 */
vos_status_t vos_thread_sleep(unsigned msec)
{
/* TODO: should change this to something like VOS_OS_HAS_NANOSLEEP */
#if defined(VOS_RTEMS) && VOS_RTEMS!=0
    enum { NANOSEC_PER_MSEC = 1000000 };
    struct timespec req;

    //VOS_CHECK_STACK;
    req.tv_sec = msec / 1000;
    req.tv_nsec = (msec % 1000) * NANOSEC_PER_MSEC;

    if (nanosleep(&req, NULL) == 0)
	return VOS_SUCCESS;

    return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
#else
    //VOS_CHECK_STACK;

    errno = VOS_SUCCESS;

    usleep(msec * 1000);

    /* MacOS X (reported on 10.5) seems to always set errno to ETIMEDOUT.
     * It does so because usleep() is declared to return int, and we're
     * supposed to check for errno only when usleep() returns non-zero. 
     * Unfortunately, usleep() is declared to return void in other platforms
     * so it's not possible to always check for the return value (unless 
     * we add a detection routine in autoconf).
     *
     * As a workaround, here we check if ETIMEDOUT is returned and
     * return successfully if it is.
     */
    if (vos_get_native_os_error() == ETIMEDOUT)
	    return VOS_SUCCESS;

    return vos_get_native_os_error();

#endif	/* VOS_RTEMS */
}

#if defined(VOS_OS_HAS_CHECK_STACK) && VOS_OS_HAS_CHECK_STACK!=0
/*
 * vos_thread_check_stack()
 * Implementation for //VOS_CHECK_STACK
 */
void vos_thread_check_stack(const char *file, int line)
{
    char stk_ptr;
    vos_uint32_t usage;
    vos_thread_t *thread = vos_thread_this();

    /* Calculate current usage. */
    usage = (&stk_ptr > thread->stk_start) ? &stk_ptr - thread->stk_start :
		thread->stk_start - &stk_ptr;

    /* Assert if stack usage is dangerously high. */
    vos_assert("STACK OVERFLOW!! " && (usage <= thread->stk_size - 128));

    /* Keep statistic. */
    if (usage > thread->stk_max_usage) {
	thread->stk_max_usage = usage;
	thread->caller_file = file;
	thread->caller_line = line;
    }
}

/*
 * vos_thread_get_stack_max_usage()
 */
vos_uint32_t vos_thread_get_stack_max_usage(vos_thread_t *thread)
{
    return thread->stk_max_usage;
}

/*
 * vos_thread_get_stack_info()
 */
vos_status_t vos_thread_get_stack_info( vos_thread_t *thread,
					      const char **file,
					      int *line )
{
    vos_assert(thread);

    *file = thread->caller_file;
    *line = thread->caller_line;
    return 0;
}

#endif	/* VOS_OS_HAS_CHECK_STACK */

///////////////////////////////////////////////////////////////////////////////
/*
 * vos_atomic_create()
 */
vos_status_t vos_atomic_create( vos_atomic_value_t initial, vos_atomic_t **ptr_atomic)
{
    vos_status_t rc;
    vos_atomic_t *atomic_var;

    atomic_var = VOS_MALLOC_T(vos_atomic_t);

    VOS_ASSERT_RETURN(atomic_var, VOS_ENOMEM);
    
#if VOS_HAS_THREADS
    rc = vos_mutex_create("atm%p", VOS_MUTEX_SIMPLE, &atomic_var->mutex);
    if (rc != VOS_SUCCESS)
	return rc;
#endif
    atomic_var->value = initial;

    *ptr_atomic = atomic_var;
    return VOS_SUCCESS;
}

/*
 * vos_atomic_destroy()
 */
vos_status_t vos_atomic_destroy( vos_atomic_t *atomic_var )
{
    VOS_ASSERT_RETURN(atomic_var, VOS_EINVAL);
#if VOS_HAS_THREADS
    return vos_mutex_destroy( atomic_var->mutex );
#else
    return 0;
#endif
}

/*
 * vos_atomic_set()
 */
void vos_atomic_set(vos_atomic_t *atomic_var, vos_atomic_value_t value)
{
//    //VOS_CHECK_STACK;

#if VOS_HAS_THREADS
    vos_mutex_lock( atomic_var->mutex );
#endif
    atomic_var->value = value;
#if VOS_HAS_THREADS
    vos_mutex_unlock( atomic_var->mutex);
#endif 
}

/*
 * vos_atomic_get()
 */
vos_atomic_value_t vos_atomic_get(vos_atomic_t *atomic_var)
{
    vos_atomic_value_t oldval;
    
 //   //VOS_CHECK_STACK;

#if VOS_HAS_THREADS
    vos_mutex_lock( atomic_var->mutex );
#endif
    oldval = atomic_var->value;
#if VOS_HAS_THREADS
    vos_mutex_unlock( atomic_var->mutex);
#endif
    return oldval;
}

/*
 * vos_atomic_inc_and_get()
 */
vos_atomic_value_t vos_atomic_inc_and_get(vos_atomic_t *atomic_var)
{
    vos_atomic_value_t new_value;

//    //VOS_CHECK_STACK;

#if VOS_HAS_THREADS
    vos_mutex_lock( atomic_var->mutex );
#endif
    new_value = ++atomic_var->value;
#if VOS_HAS_THREADS
    vos_mutex_unlock( atomic_var->mutex);
#endif

    return new_value;
}
/*
 * vos_atomic_inc()
 */
void vos_atomic_inc(vos_atomic_t *atomic_var)
{
    vos_atomic_inc_and_get(atomic_var);
}

/*
 * vos_atomic_dec_and_get()
 */
vos_atomic_value_t vos_atomic_dec_and_get(vos_atomic_t *atomic_var)
{
    vos_atomic_value_t new_value;

//    //VOS_CHECK_STACK;

#if VOS_HAS_THREADS
    vos_mutex_lock( atomic_var->mutex );
#endif
    new_value = --atomic_var->value;
#if VOS_HAS_THREADS
    vos_mutex_unlock( atomic_var->mutex);
#endif

    return new_value;
}

/*
 * vos_atomic_dec()
 */
void vos_atomic_dec(vos_atomic_t *atomic_var)
{
    vos_atomic_dec_and_get(atomic_var);
}

/*
 * vos_atomic_add_and_get()
 */ 
vos_atomic_value_t vos_atomic_add_and_get( vos_atomic_t *atomic_var, 
                                                 vos_atomic_value_t value )
{
    vos_atomic_value_t new_value;

#if VOS_HAS_THREADS
    vos_mutex_lock(atomic_var->mutex);
#endif
    
    atomic_var->value += value;
    new_value = atomic_var->value;

#if VOS_HAS_THREADS
    vos_mutex_unlock(atomic_var->mutex);
#endif

    return new_value;
}

/*
 * vos_atomic_add()
 */ 
void vos_atomic_add( vos_atomic_t *atomic_var, 
                            vos_atomic_value_t value )
{
    vos_atomic_add_and_get(atomic_var, value);
}

///////////////////////////////////////////////////////////////////////////////
/*
 * vos_thread_local_alloc()
 */
vos_status_t vos_thread_local_alloc(long *p_index)
{
#if VOS_HAS_THREADS
    pthread_key_t key;
    int rc;

    VOS_ASSERT_RETURN(p_index != NULL, VOS_EINVAL);

    vos_assert( sizeof(pthread_key_t) <= sizeof(long));
    if ((rc=pthread_key_create(&key, NULL)) != 0)
    {
	    return VOS_RETURN_OS_ERROR(rc);
    }
    *p_index = key;
    return VOS_SUCCESS;
#else
    int i;
    for (i=0; i<MAX_THREADS; ++i) 
    {
    	if (tls_flag[i] == 0)
    	{
    	    break;
    	}
    }
    if (i == MAX_THREADS) 
	return VOS_ETOOMANY;
    
    tls_flag[i] = 1;
    tls[i] = NULL;

    *p_index = i;
    return VOS_SUCCESS;
#endif
}

/*
 * vos_thread_local_free()
 */
void vos_thread_local_free(long index)
{
//    //VOS_CHECK_STACK;
#if VOS_HAS_THREADS
    pthread_key_delete(index);
#else
    tls_flag[index] = 0;
#endif
}

/*
 * vos_thread_local_set()
 */
vos_status_t vos_thread_local_set(long index, void *value)
{
    //Can't check stack because this function is called in the
    //beginning before main thread is initialized.
    ////VOS_CHECK_STACK;
#if VOS_HAS_THREADS
    int rc=pthread_setspecific(index, value);
    return rc==0 ? VOS_SUCCESS : VOS_RETURN_OS_ERROR(rc);
#else
    vos_assert(index >= 0 && index < MAX_THREADS);
    tls[index] = value;
    return VOS_SUCCESS;
#endif
}

void* vos_thread_local_get(long index)
{
    //Can't check stack because this function is called
    //by //VOS_CHECK_STACK itself!!!
    ////VOS_CHECK_STACK;
#if VOS_HAS_THREADS
    return pthread_getspecific(index);
#else
    vos_assert(index >= 0 && index < MAX_THREADS);
    return tls[index];
#endif
}

///////////////////////////////////////////////////////////////////////////////
void vos_enter_critical_section(void)
{
#if VOS_HAS_THREADS
    vos_mutex_lock(&critical_section);
#endif
}

void vos_leave_critical_section(void)
{
#if VOS_HAS_THREADS
    vos_mutex_unlock(&critical_section);
#endif
}


///////////////////////////////////////////////////////////////////////////////
#if defined(VOS_LINUX) && VOS_LINUX!=0
VOS_BEGIN_DECL
int pthread_mutexattr_settype(pthread_mutexattr_t*,int);
VOS_END_DECL
#endif

static vos_status_t init_mutex(vos_mutex_t *mutex, const char *name, int type)
{
#if VOS_HAS_THREADS
    pthread_mutexattr_t attr;
    int rc;

//    //VOS_CHECK_STACK;

    rc = pthread_mutexattr_init(&attr);
    if (rc != 0)
	return VOS_RETURN_OS_ERROR(rc);

    if (type == VOS_MUTEX_SIMPLE) 
    {
        #if (defined(VOS_LINUX) && VOS_LINUX!=0) || defined(VOS_HAS_PTHREAD_MUTEXATTR_SETTYPE)
    	rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP);
        #else
    	rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
        #endif
    } 
    else 
    {
        #if (defined(VOS_LINUX) && VOS_LINUX!=0) || defined(VOS_HAS_PTHREAD_MUTEXATTR_SETTYPE)
    	rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
        #else
    	rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        #endif
    }
    
    if (rc != 0) 
    {
	    return VOS_RETURN_OS_ERROR(rc);
    }

    rc = pthread_mutex_init(&mutex->mutex, &attr);
    if (rc != 0) 
    {
	    return VOS_RETURN_OS_ERROR(rc);
    }
    
    rc = pthread_mutexattr_destroy(&attr);
    if (rc != 0) 
    {
    	vos_status_t status = VOS_RETURN_OS_ERROR(rc);
    	pthread_mutex_destroy(&mutex->mutex);
    	return status;
    }

#if VOS_DEBUG
    /* Set owner. */
    mutex->nesting_level = 0;
    mutex->owner = NULL;
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
    	strncpy(mutex->obj_name, name, VOS_MAX_OBJ_NAME);
    	mutex->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    VOS_LOG(6, (mutex->obj_name, "Mutex created"));
    return VOS_SUCCESS;
#else /* VOS_HAS_THREADS */
    return VOS_SUCCESS;
#endif
}

/*
 * vos_mutex_create()
 */
vos_status_t vos_mutex_create(const char *name, 
				    int type,
				    vos_mutex_t **ptr_mutex)
{
#if VOS_HAS_THREADS
    vos_status_t rc;
    vos_mutex_t *mutex;

    VOS_ASSERT_RETURN(ptr_mutex, VOS_EINVAL);

    mutex = VOS_MALLOC_T(vos_mutex_t);
    VOS_ASSERT_RETURN(mutex, VOS_ENOMEM);

    if ((rc=init_mutex(mutex, name, type)) != VOS_SUCCESS)
    {
	    return rc;
    }
    *ptr_mutex = mutex;
    return VOS_SUCCESS;
#else /* VOS_HAS_THREADS */
    *ptr_mutex = (vos_mutex_t*)1;
    return VOS_SUCCESS;
#endif
}

/*
 * vos_mutex_create_simple()
 */
vos_status_t vos_mutex_create_simple(const char *name, vos_mutex_t **mutex )
{
    return vos_mutex_create(name, VOS_MUTEX_SIMPLE, mutex);
}

/*
 * vos_mutex_create_recursive()
 */
vos_status_t vos_mutex_create_recursive(  const char *name, vos_mutex_t **mutex )
{
    return vos_mutex_create(name, VOS_MUTEX_RECURSE, mutex);
}

/*
 * vos_mutex_lock()
 */
vos_status_t vos_mutex_lock(vos_mutex_t *mutex)
{
#if VOS_HAS_THREADS
    vos_status_t status;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

#if VOS_DEBUG
    VOS_LOG(6,(mutex->obj_name, "Mutex: thread %s is waiting (mutex owner=%s)", 
				vos_thread_this()->obj_name,
				mutex->owner_name));
#else
    VOS_LOG(6,(mutex->obj_name, "Mutex: thread %s is waiting", 
				vos_thread_this()->obj_name));
#endif

    status = pthread_mutex_lock( &mutex->mutex );


#if VOS_DEBUG
    if (status == VOS_SUCCESS) 
    {
    	mutex->owner = vos_thread_this();
    	vos_ansi_strcpy(mutex->owner_name, mutex->owner->obj_name);
    	++mutex->nesting_level;
    }

    VOS_LOG(6,(mutex->obj_name, 
	      (status==0 ? 
		"Mutex acquired by thread %s (level=%d)" : 
		"Mutex acquisition FAILED by %s (level=%d)"),
	      vos_thread_this()->obj_name,
	      mutex->nesting_level));
#else
    VOS_LOG(6,(mutex->obj_name, 
	      (status==0 ? "Mutex acquired by thread %s" : "FAILED by %s"),
	      vos_thread_this()->obj_name));
#endif

    if (status == 0)
    {
	    return VOS_SUCCESS;
    }
    
	return VOS_RETURN_OS_ERROR(status);
#else	/* VOS_HAS_THREADS */
    vos_assert( mutex == (vos_mutex_t*)1 );
    return VOS_SUCCESS;
#endif
}

/*
 * vos_mutex_unlock()
 */
vos_status_t vos_mutex_unlock(vos_mutex_t *mutex)
{
#if VOS_HAS_THREADS
    vos_status_t status;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

#if VOS_DEBUG
    vos_assert(mutex->owner == vos_thread_this());
    if (--mutex->nesting_level == 0) 
    {
    	mutex->owner = NULL;
    	mutex->owner_name[0] = '\0';
    }

    VOS_LOG(6,(mutex->obj_name, "Mutex released by thread %s (level=%d)", 
				vos_thread_this()->obj_name, 
				mutex->nesting_level));
#else
    VOS_LOG(6,(mutex->obj_name, "Mutex released by thread %s", 
				vos_thread_this()->obj_name));
#endif

    status = pthread_mutex_unlock( &mutex->mutex );
    if (status == 0)
    {
	    return VOS_SUCCESS;
    }
    
	return VOS_RETURN_OS_ERROR(status);

#else /* VOS_HAS_THREADS */
    vos_assert( mutex == (vos_mutex_t*)1 );
    return VOS_SUCCESS;
#endif
}

/*
 * mutex_trylock()
 */
vos_status_t vos_mutex_trylock(vos_mutex_t *mutex)
{
#if VOS_HAS_THREADS
    int status;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

    VOS_LOG(6,(mutex->obj_name, "Mutex: thread %s is trying", 
				vos_thread_this()->obj_name));

    status = pthread_mutex_trylock( &mutex->mutex );

    if (status==0) 
    {
        #if VOS_DEBUG
    	mutex->owner = vos_thread_this();
    	vos_ansi_strcpy(mutex->owner_name, mutex->owner->obj_name);
    	++mutex->nesting_level;

    	VOS_LOG(6,(mutex->obj_name, "Mutex acquired by thread %s (level=%d)", 
    				   vos_thread_this()->obj_name,
    				   mutex->nesting_level));
        #else
    	VOS_LOG(6,(mutex->obj_name, "Mutex acquired by thread %s", 
    				  vos_thread_this()->obj_name));
        #endif
    } 
    else 
    {
    	VOS_LOG(6,(mutex->obj_name, "Mutex: thread %s's trylock() failed", 
    				    vos_thread_this()->obj_name));
    }
    
    if (status==0)
    {
	    return VOS_SUCCESS;
    }
	return VOS_RETURN_OS_ERROR(status);
#else	/* VOS_HAS_THREADS */
    vos_assert( mutex == (vos_mutex_t*)1);
    return VOS_SUCCESS;
#endif
}

/*
 * vos_mutex_destroy()
 */
vos_status_t vos_mutex_destroy(vos_mutex_t *mutex)
{
    enum { RETRY = 4 };
    int status = 0;
    unsigned retry;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(mutex, VOS_EINVAL);

#if VOS_HAS_THREADS
    VOS_LOG(6,(mutex->obj_name, "Mutex destroyed by thread %s",
			       vos_thread_this()->obj_name));

    for (retry=0; retry<RETRY; ++retry) 
    {
    	status = pthread_mutex_destroy( &mutex->mutex );
    	if (status == VOS_SUCCESS)
    	{
    	    break;
        }
    	else if (retry<RETRY-1 && status == EBUSY)
    	{
    	    pthread_mutex_unlock(&mutex->mutex);
        }
    }

    if (status == 0)
    {
	    return VOS_SUCCESS;
    }
	return VOS_RETURN_OS_ERROR(status);
    
#else
    vos_assert( mutex == (vos_mutex_t*)1 );
    status = VOS_SUCCESS;
    return status;
#endif
}

#if VOS_DEBUG
vos_bool_t vos_mutex_is_locked(vos_mutex_t *mutex)
{
#if VOS_HAS_THREADS
    return mutex->owner == vos_thread_this();
#else
    return 1;
#endif
}
#endif


#if defined(VOS_HAS_SEMAPHORE) && VOS_HAS_SEMAPHORE != 0
vos_status_t vos_sem_create(const char *name,
				   unsigned initial, 
				   unsigned max,
				   vos_sem_t **ptr_sem)
{
#if VOS_HAS_THREADS
    vos_sem_t *sem;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN( ptr_sem != NULL, VOS_EINVAL);

    sem = VOS_MALLOC_T(vos_sem_t);
    VOS_ASSERT_RETURN(sem, VOS_ENOMEM);

#if defined(VOS_DARWINOS) && VOS_DARWINOS!=0
    /* MacOS X doesn't support anonymous semaphore */
    {
    	char sem_name[VOS_GUID_MAX_LENGTH+1];
    	vos_str_t nam;

    	/* We should use SEM_NAME_LEN, but this doesn't seem to be 
    	 * declared anywhere? The value here is just from trial and error
    	 * to get the longest name supported.
    	 */
        #define MAX_SEM_NAME_LEN	23

    	/* Create a unique name for the semaphore. */
    	if (VOS_GUID_STRING_LENGTH <= MAX_SEM_NAME_LEN) 
    	{
    	    nam.ptr = sem_name;
    	    vos_generate_unique_string(&nam);
    	    sem_name[nam.slen] = '\0';
    	} 
    	else 
    	{
    	    vos_create_random_string(sem_name, MAX_SEM_NAME_LEN);
    	    sem_name[MAX_SEM_NAME_LEN] = '\0';
    	}

    	/* Create semaphore */
    	sem->sem = sem_open(sem_name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, 
    			    initial);
    	if (sem->sem == SEM_FAILED)
    	    return VOS_RETURN_OS_ERROR(vos_get_native_os_error());

    	/* And immediately release the name as we don't need it */
    	sem_unlink(sem_name);
    }
#else
    sem->sem = VOS_MALLOC_T(sem_t);
    if (sem_init( sem->sem, 0, initial) != 0) 
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
	}
#endif
    
    /* Set name. */
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
    	strncpy(sem->obj_name, name, VOS_MAX_OBJ_NAME);
    	sem->obj_name[VOS_MAX_OBJ_NAME-1] = '\0';
    }

    VOS_LOG(6, (sem->obj_name, "Semaphore created"));

    *ptr_sem = sem;
    return VOS_SUCCESS;
#else
    *ptr_sem = (vos_sem_t*)1;
    return VOS_SUCCESS;
#endif
}

/*
 * vos_sem_wait()
 */
vos_status_t vos_sem_wait(vos_sem_t *sem)
{
#if VOS_HAS_THREADS
    int result;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    VOS_LOG(6, (sem->obj_name, "Semaphore: thread %s is waiting", 
			      vos_thread_this()->obj_name));

    result = sem_wait( sem->sem );
    
    if (result == 0) 
    {
    	VOS_LOG(6, (sem->obj_name, "Semaphore acquired by thread %s", 
    				  vos_thread_this()->obj_name));
    } 
    else 
    {
    	VOS_LOG(6, (sem->obj_name, "Semaphore: thread %s FAILED to acquire", 
    				  vos_thread_this()->obj_name));
    }

    if (result == 0)
    {
	    return VOS_SUCCESS;
    }
	return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
#else
    vos_assert( sem == (vos_sem_t*) 1 );
    return VOS_SUCCESS;
#endif
}

/*
 * vos_vos_sem_trywait()
 */
vos_status_t vos_sem_trywait(vos_sem_t *sem)
{
#if VOS_HAS_THREADS
    int result;

//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    result = sem_trywait( sem->sem );
    
    if (result == 0) 
    {
    	VOS_LOG(6, (sem->obj_name, "Semaphore acquired by thread %s", 
    				  vos_thread_this()->obj_name));
    } 
    
    if (result == 0)
    {
	    return VOS_SUCCESS;
    }
    
	return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
#else
    vos_assert( sem == (vos_sem_t*)1 );
    return VOS_SUCCESS;
#endif
}

/*
 * vos_sem_post()
 */
vos_status_t vos_sem_post(vos_sem_t *sem)
{
#if VOS_HAS_THREADS
    int result;
    VOS_LOG(6, (sem->obj_name, "Semaphore released by thread %s",
			      vos_thread_this()->obj_name));
    result = sem_post( sem->sem );

    if (result == 0)
    {
	    return VOS_SUCCESS;
    }
	return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
#else
    vos_assert( sem == (vos_sem_t*) 1);
    return VOS_SUCCESS;
#endif
}

/*
 * vos_sem_destroy()
 */
vos_status_t vos_sem_destroy(vos_sem_t *sem)
{
#if VOS_HAS_THREADS
    int result;

 //   //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(sem, VOS_EINVAL);

    VOS_LOG(6, (sem->obj_name, "Semaphore destroyed by thread %s",
			      vos_thread_this()->obj_name));
#if defined(VOS_DARWINOS) && VOS_DARWINOS!=0
    result = sem_close( sem->sem );
#else
    result = sem_destroy( sem->sem );
#endif

    if (result == 0)
    {
	    return VOS_SUCCESS;
    }
	return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
#else
    vos_assert( sem == (vos_sem_t*) 1 );
    return VOS_SUCCESS;
#endif
}

#endif	/* VOS_HAS_SEMAPHORE */

///////////////////////////////////////////////////////////////////////////////
#if defined(VOS_HAS_EVENT_OBJ) && VOS_HAS_EVENT_OBJ != 0

/*
 * vos_event_create()
 */
vos_status_t vos_event_create(const char *name,
				    vos_bool_t manual_reset, vos_bool_t initial,
				    vos_event_t **ptr_event)
{
    vos_assert(!"Not supported!");
    VOS_UNUSED_ARG(name);
    VOS_UNUSED_ARG(manual_reset);
    VOS_UNUSED_ARG(initial);
    VOS_UNUSED_ARG(ptr_event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_wait()
 */
vos_status_t vos_event_wait(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_trywait()
 */
vos_status_t vos_event_trywait(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_set()
 */
vos_status_t vos_event_set(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_pulse()
 */
vos_status_t vos_event_pulse(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_reset()
 */
vos_status_t vos_event_reset(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

/*
 * vos_event_destroy()
 */
vos_status_t vos_event_destroy(vos_event_t *event)
{
    VOS_UNUSED_ARG(event);
    return VOS_EINVALIDOP;
}

#endif	/* VOS_HAS_EVENT_OBJ */

///////////////////////////////////////////////////////////////////////////////
#if defined(VOS_TERM_HAS_COLOR) && VOS_TERM_HAS_COLOR != 0
/*
 * Terminal
 */

/**
 * Set terminal color.
 */
vos_status_t vos_term_set_color(vos_color_t color)
{
    /* put bright prefix to ansi_color */
    char ansi_color[12] = "\033[01;3";

    if (color & VOS_TERM_COLOR_BRIGHT) {
	color ^= VOS_TERM_COLOR_BRIGHT;
    } else {
	strcpy(ansi_color, "\033[00;3");
    }

    switch (color) 
    {
    case 0:
    	/* black color */
    	strcat(ansi_color, "0m");
    	break;
    case VOS_TERM_COLOR_R:
    	/* red color */
    	strcat(ansi_color, "1m");
    	break;
    case VOS_TERM_COLOR_G:
    	/* green color */
    	strcat(ansi_color, "2m");
    	break;
    case VOS_TERM_COLOR_B:
    	/* blue color */
    	strcat(ansi_color, "4m");
    	break;
    case VOS_TERM_COLOR_R | VOS_TERM_COLOR_G:
    	/* yellow color */
    	strcat(ansi_color, "3m");
    	break;
    case VOS_TERM_COLOR_R | VOS_TERM_COLOR_B:
    	/* magenta color */
    	strcat(ansi_color, "5m");
    	break;
    case VOS_TERM_COLOR_G | VOS_TERM_COLOR_B:
    	/* cyan color */
    	strcat(ansi_color, "6m");
    	break;
    case VOS_TERM_COLOR_R | VOS_TERM_COLOR_G | VOS_TERM_COLOR_B:
    	/* white color */
    	strcat(ansi_color, "7m");
    	break;
    default:
    	/* default console color */
    	strcpy(ansi_color, "\033[00m");
    	break;
    }

    fputs(ansi_color, stdout);

    return VOS_SUCCESS;
}

/**
 * Get current terminal foreground color.
 */
vos_color_t vos_term_get_color(void)
{
    return 0;
}

#endif	/* VOS_TERM_HAS_COLOR */

#if !defined(VOS_DARWINOS) || VOS_DARWINOS == 0
/*
 * vos_run_app()
 */
int vos_run_app(vos_main_func_ptr main_func, int argc, char *argv[],
                       unsigned flags)
{
    return (*main_func)(argc, argv);
}
#endif

#endif

