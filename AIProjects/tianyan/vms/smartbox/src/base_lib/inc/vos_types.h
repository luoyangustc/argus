
#ifndef __VOS_TYPES_H__
#define __VOS_TYPES_H__

#include    <stdio.h>
#include    <string.h>
#include    <stdlib.h>
#include    "vos_config.h"

#if (OS_UCOS_II == 1)
#include    "ucos_ii.h"
#include    "mymem.h"
#include    "inet.h"
#include    "sockets.h"
#include    "errno.h"
#include    "printf.h"
#include    "acw.h"
#endif

#undef  EXT
#ifdef __TYPES_C__
#define EXT 
#else
#define EXT extern
#endif

#ifdef __cplusplus
#  define VOS_BEGIN_DECL		    extern "C" {
#  define VOS_END_DECL		    }
#else
#  define VOS_BEGIN_DECL
#  define VOS_END_DECL
#endif

VOS_BEGIN_DECL

#if (OS_UCOS_II==1)
    typedef int								vos_size_t;
#else
    typedef unsigned int					vos_size_t;
#endif

typedef long long						vos_int64_t;
typedef unsigned long long				vos_uint64_t;
typedef int								vos_int32_t;
typedef unsigned int					vos_uint32_t;
typedef short							vos_int16_t;
typedef unsigned short					vos_uint16_t;
typedef signed char						vos_int8_t;
typedef unsigned char					vos_uint8_t;
typedef int								vos_status_t;
typedef int								vos_bool_t;
typedef char							vos_char_t;

typedef char  							int8;
typedef short 							int16;
typedef int   							int32;
typedef unsigned char 					uint8;
typedef unsigned short 					uint16;
typedef unsigned int  					uint32;

typedef long							vos_ssize_t;
typedef vos_int64_t						vos_highprec_t;
typedef unsigned int					UINT;
typedef unsigned short 					version_t[4];

//#define vos_fd_set_t					fd_set

#if defined(VOS_HAS_INT64) && VOS_HAS_INT64!=0
typedef vos_int64_t                     vos_offset_t;
#else
typedef vos_ssize_t                     vos_offset_t;
#endif



#ifndef uint64
typedef unsigned long long 			uint64;
#endif

#ifndef DWORD
#ifdef _MSC_VER
typedef unsigned long			DWORD;
#else
typedef unsigned int			DWORD;
#endif
#endif

#ifndef WORD
typedef unsigned short				WORD;
#endif

#ifndef BOOL
typedef int							BOOL;
#endif

#ifndef BYTE
typedef unsigned char      			BYTE;
#endif

#ifndef PBYTE
typedef BYTE*		    			PBYTE;
#endif

#define VOS_SUCCESS						0

#ifndef TRUE
#define TRUE						1
#endif

#ifndef FALSE
#define FALSE						0
#endif

#ifndef true
#define true						1
#endif

#ifndef false
#define false						0
#endif

#ifndef NULL
#define NULL 						((void*)0)
#endif

#ifndef null
#define null 						((void*)0)
#endif

#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef HANDLE
typedef void *						HANDLE;
#endif

typedef struct vos_str_t vos_str_t;

/** 
* List
*/
typedef void							vos_list_type_t;
typedef struct vos_list_t				vos_list_t;


/** 
* Hash
*/
typedef struct vos_hash_table_t			vos_hash_table_t;
typedef struct vos_hash_entry_t			vos_hash_entry_t;
typedef struct vos_hash_iterator_t
{
    vos_uint32_t						index;
    vos_hash_entry_t					*entry;
} vos_hash_iterator_t;

/** 
* Io queue
*/
typedef struct vos_ioqueue_t			vos_ioqueue_t;
typedef struct vos_ioqueue_key_t		vos_ioqueue_key_t;

/** 
* Atomic
*/
typedef struct vos_atomic_t vos_atomic_t;
typedef ATOMIC_VALUE_TYPE vos_atomic_value_t;

/** 
* Thread
*/
typedef struct vos_thread_t				vos_thread_t;


/** 
* Lock
*/
typedef struct vos_lock_t				vos_lock_t;
typedef struct vos_mutex_t				vos_mutex_t;
typedef struct vos_sem_t				vos_sem_t;
typedef struct vos_event_t				vos_event_t;
typedef struct vos_pipe_t				vos_pipe_t;


///////////////////////////////////////////////////////

//net

///////////////////////////////////////////////////////
#define VOS_INADDR_ANY					((vos_uint32_t)0)
#define VOS_INADDR_NONE					((vos_uint32_t)0xffffffff)
#define VOS_INADDR_BROADCAST			((vos_uint32_t)0xffffffff)
#define VOS_INVALID_SOCKET				(-1)

typedef void                            vos_sockaddr_t;
typedef long							vos_sock_t;
typedef struct vos_sockaddr_in          vos_sockaddr_in;
typedef int								vos_socklen_t;


/** 
* 
*/
typedef void							*vos_oshandle_t;
typedef unsigned int					vos_color_t;
typedef int								vos_exception_id_t;


///////////////////////////////////////////////////////

//mem

///////////////////////////////////////////////////////
typedef struct vos_pool_t				vos_pool_t;
#define VOS_ARRAY_SIZE(a)				(sizeof(a)/sizeof(a[0]))
#define VOS_MAXINT32					0x7FFFFFFFL
#define VOS_MAX_OBJ_NAME				32


///////////////////////////////////////////////////////

//time

///////////////////////////////////////////////////////
typedef struct vos_timer_heap_t			vos_timer_heap_t;

typedef struct vos_time_val
{
    long								sec;
    long								usec;

} vos_time_val;

typedef struct vos_parsed_time
{
    int									wday;   /*0~6(0:Sunday)*/
    int									day;    /*1-31 */
    int									mon;    /*0 - 11*/
    int									year;
    int									sec;
    int									min;
    int									hour;   /*0~23*/
    int									msec;   /*0-999 */

} vos_parsed_time;

typedef union vos_timestamp_t
{
    struct
    {
#if defined(_M_IS_LITTLE_ENDIAN) && _M_IS_LITTLE_ENDIAN!=0
        vos_uint32_t lo;
        vos_uint32_t hi;
#else
        vos_uint32_t hi;
        vos_uint32_t lo;
#endif
    } u32; /* The 64-bit value as two 32-bit values. */

#if VOS_HAS_INT64
    vos_uint64_t u64;
#endif
} vos_timestamp_t;

#define VOS_UNUSER(var)					(var=var)

#define VOS_MALLOC_T(type)				((type*)malloc(sizeof(type)))//((type*)appmem_malloc_debug(__FILE__, __LINE__, sizeof(type)))
#define VOS_MALLOC_BLK_T(type, n)		((type*)malloc(sizeof(type) * (n)))//((type*)appmem_malloc_debug(__FILE__, __LINE__, sizeof(type)*(n)))
#define VOS_CALLOC_T(n,type)            ((type*)calloc(n, sizeof(type)))
#define VOS_CALLOC_BLK_T(type, n)       ((type*)calloc(n, sizeof(type) * (n)))
#define VOS_FREE_T(p)					(free(p))//(appmem_free(p))

#if (OS_WIN32 == 1)
#define VOS_INLINE					__inline
#elif (OS_LINUX == 1)
#define VOS_INLINE					inline
#elif (OS_UCOS_II == 1)
#define VOS_INLINE					_Inline
#else
#define VOS_INLINE					inline
#endif

//printf

///////////////////////////////////////////////////////
#define VOS_PRINTF						printf//fhprintf
#define uprintf							printf
#if (OS_UCOS_II == 1)
#else
#define fhprintf						printf
#endif


///////////////////////////////////////////////////////

//error

///////////////////////////////////////////////////////
#if (OS_WIN32 == 1)
#include <WinSock2.h>
#define VOS_EINPROGRESS					WSAEINPROGRESS
#define VOS_EWOULDBLOCK					WSAEWOULDBLOCK
#define VOS_EINTR	                    WSAEINTR	
#else
#define VOS_EINPROGRESS					EINPROGRESS
#define VOS_EWOULDBLOCK					EWOULDBLOCK
#define VOS_EINTR	                    EINTR
#endif

enum
{
    VOS_EUNKNOWN = 1000,
    VOS_EPENDING,

    /**
    * sockets连接太多
    */
    VOS_ETOOMANYCONN,

    /**
    * 无效参数
    */
    VOS_EINVAL,

    /**
    * 名称太长(eg. hostname)
    */
    VOS_ENAMETOOLONG,

    /**
    * Not found
    */
    VOS_ENOTFOUND,

    /**
    * Not enough memory
    */
    VOS_ENOMEM,

    /**
    * Bug detected
    */
    VOS_EBUG,

    /**
    * 超时
    */
    VOS_ETIMEDOUT,

    /**
    * Too many objects.
    */
    VOS_ETOOMANY,

    /**
    * Busy.
    */
    VOS_EBUSY,

    /**
    * Not supported
    */
    VOS_ENOTSUP,

    /**
    * Invalid operation
    */
    VOS_EINVALIDOP,

    /**
    * Operation is cancelled
    */
    VOS_ECANCELLED,

    /**
    * Object already exists
    */
    VOS_EEXISTS,

    /**
    * End of file
    */
    VOS_EEOF,

    /**
    * Size is too big
    */
    VOS_ETOOBIG,

    /**
    * gethostbyname()函数调用失败返回的错误
    */
    VOS_ERESOLVE,

    /**
    * Size is too small.
    */
    VOS_ETOOSMALL,

    /**
    * Ignored
    */
    VOS_EIGNORED,

    /**
    * IPv6 is not supported
    */
    VOS_EIPV6NOTSUP,

    /**
    * Unsupported address family
    */
    VOS_EAFNOTSUP,

    /**
    * Object no longer exists
    */
    VOS_EGONE,
    /**
    * Socket is stopped
    */
    VOS_ESOCKETSTOP,
    /**
    * BUFFER溢出
    */
    VOS_EBUFFOVERFLOW,
};

#ifndef VOS_RETURN_OS_ERROR
#define VOS_RETURN_OS_ERROR(os_code)   ((os_code) ? (os_code) : -1)//(os_code ? VOS_STATUS_FROM_OS(os_code) : -1)
#endif

VOS_END_DECL

#endif /* __VOS_TYPES_H__ */