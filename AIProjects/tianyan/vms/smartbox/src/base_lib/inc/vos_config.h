#ifndef __VOS_CONFIG_H__
#define __VOS_CONFIG_H__

#undef	EXT
#ifndef __OS_CORE_C__
#define EXT extern
#else
#define EXT
#endif

//#define OS_WIN32    1
//#define OS_LINUX    0
//#define OS_UCOS_II   0

#define VOS_VERSION_NUM_REV	            1
#define VOS_VERSION_NUM_MAJOR	        1
#define VOS_VERSION_NUM_MINOR	        0
#define VOS_VERSION_NUM_EXTRA	        ""

#if (OS_WIN32 == 1)
#include "vos_config_win32.h"
#elif (OS_LINUX == 1)
#include "vos_config_unix.h"
#else
//config_xx.h
#endif

//#define ATOMIC_VALUE_TYPE   long

////////////////////////////////////////////////////
#define VOS_UNUSED_ARG(arg)             (void)arg

////////////////////////////////////////////////////

//log

////////////////////////////////////////////////////
#ifndef VOS_LOG_MAX_LEVEL
#  define VOS_LOG_MAX_LEVEL   5
#endif

#ifndef VOS_LOG_MAX_SIZE
#  define VOS_LOG_MAX_SIZE	    4000
#endif

#ifndef VOS_LOG_INDENT_SIZE
#   define VOS_LOG_INDENT_SIZE        1
#endif

#ifndef VOS_THREAD_DEFAULT_STACK_SIZE 
#  define VOS_THREAD_DEFAULT_STACK_SIZE    8192
#endif

#ifdef __cplusplus
#  define VOS_DECL_NO_RETURN(type)   VOS_DECL(type) VOS_NORETURN
#  define VOS_IDECL_NO_RETURN(type)  VOS_INLINE(type) VOS_NORETURN
#  define VOS_BEGIN_DECL		    extern "C" {
#  define VOS_END_DECL		    }
#else
#  define VOS_DECL_NO_RETURN(type)   VOS_NORETURN VOS_DECL(type)
#  define VOS_IDECL_NO_RETURN(type)  VOS_NORETURN VOS_INLINE(type)
#  define VOS_BEGIN_DECL
#  define VOS_END_DECL
#endif

#define APP_MYUART_PRIO             43
////////////////////////////////////////////////////

#ifndef VOS_SOCK_MAX_HANDLES
#   define VOS_SOCK_MAX_HANDLES 	(64)
#endif

#ifndef VOS_MAXPATH
#   define VOS_MAXPATH		            260
#endif

#ifndef VOS_MAX_HOSTNAME
#  define VOS_MAX_HOSTNAME	            (128)
#endif

EXT const char* VOS_VERSION;
EXT const char* vos_get_version(void);
EXT void vos_dump_config(void);

#endif	/* __VOS_CONFIG_H__ */


