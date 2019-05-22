
#ifndef __VOS_COMPAT_OS_LINUX_H__
#define __VOS_COMPAT_OS_LINUX_H__

#include <errno.h>

#define VOS_OS_NAME		    "linux"

#define VOS_HAS_ARPA_INET_H	    1
#define VOS_HAS_ASSERT_H		    1
#define VOS_HAS_CTYPE_H		    1
#define VOS_HAS_ERRNO_H		    1
#define VOS_HAS_LINUX_SOCKET_H	    0
#define VOS_HAS_MALLOC_H		    1
#define VOS_HAS_NETDB_H		    1
#define VOS_HAS_NETINET_IN_H	    1
#define VOS_HAS_SETJMP_H		    1
#define VOS_HAS_STDARG_H		    1
#define VOS_HAS_STDDEF_H		    1
#define VOS_HAS_STDIO_H		    1
#define VOS_HAS_STDLIB_H		    1
#define VOS_HAS_SYS_IOCTL_H	    1
#define VOS_HAS_SYS_SELECT_H	    1
#define VOS_HAS_SYS_SOCKET_H	    1
#define VOS_HAS_SYS_TIME_H	    1
#define VOS_HAS_SYS_TIMEB_H	    1
#define VOS_HAS_SYS_TYPES_H	    1
#define VOS_HAS_TIME_H		    1
#define VOS_HAS_UNISTD_H		    1
#define VOS_HAS_SEMAPHORE_H	    1

#define VOS_HAS_MSWSOCK_H	    0
#define VOS_HAS_WINSOCK_H	    0
#define VOS_HAS_WINSOCK2_H	    0

#define VOS_SOCK_HAS_INET_ATON	    1

/* sockaddr_in中是否含有sin_len成员. 
 * Default: 0
 */
#define VOS_SOCKADDR_HAS_LEN	    0


/* errno是否是系统错误
 */
#define VOS_HAS_ERRNO_VAR	    1


#define VOS_HAS_HIGH_RES_TIMER	    1
#define VOS_HAS_MALLOC               1
#ifndef VOS_OS_HAS_CHECK_STACK
#   define VOS_OS_HAS_CHECK_STACK    0
#endif
#define VOS_NATIVE_STRING_IS_UNICODE 0

#define VOS_HAS_INT64	                1

#define ATOMIC_VALUE_TYPE	    long

/* Linux has socklen_t */
#define VOS_HAS_SOCKLEN_T		1
#define VOS_HAS_THREADS	    1

#define VOS_INLINE_SPECIFIER inline

#define vos_get_native_os_error()	    (errno)

#endif	/* __VOS_COMPAT_OS_LINUX_H__ */


