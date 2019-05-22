#ifndef __VOS_COMPAT_SOCKET_H__
#define __VOS_COMPAT_SOCKET_H__

#include "vos_config.h"

#if defined(VOS_HAS_WINSOCK2_H) && VOS_HAS_WINSOCK2_H != 0
#include <winsock2.h>
#endif

#if defined(VOS_HAS_WINSOCK_H) && VOS_HAS_WINSOCK_H != 0
#include <winsock.h>
#endif

#if defined(VOS_HAS_WS2TCPIP_H) && VOS_HAS_WS2TCPIP_H != 0
#include <ws2tcpip.h>
#endif


#if defined(_MSC_VER) && defined(VOS_HAS_IPV6) && VOS_HAS_IPV6!=0
#ifndef s_addr
#define s_addr  S_un.S_addr
#endif

#if !defined(IPPROTO_IPV6)
#include <tpipv6.h>
#endif

#define VOS_SOCK_HAS_GETADDRINFO      1
#endif	/* _MSC_VER */

#if defined(VOS_HAS_SYS_TYPES_H) && VOS_HAS_SYS_TYPES_H != 0
#include <sys/types.h>
#endif

#if defined(VOS_HAS_SYS_SOCKET_H) && VOS_HAS_SYS_SOCKET_H != 0
#include <sys/socket.h>
#endif

#if defined(VOS_HAS_SYS_SELECT_H) && VOS_HAS_SYS_SELECT_H != 0
#include <sys/select.h>
#endif

#if defined(VOS_HAS_NETINET_IN_H) && VOS_HAS_NETINET_IN_H != 0
#include <netinet/in.h>
#endif


#if defined(VOS_HAS_NETINET_TCP_H) && VOS_HAS_NETINET_TCP_H != 0
/* To pull in TCP_NODELAY constants */
#include <netinet/tcp.h>
#endif

#if defined(VOS_HAS_ARPA_INET_H) && VOS_HAS_ARPA_INET_H != 0
#include <arpa/inet.h>
#endif

#if defined(VOS_HAS_SYS_IOCTL_H) && VOS_HAS_SYS_IOCTL_H != 0
#include <sys/ioctl.h>	/* FBIONBIO */
#endif

#if defined(VOS_HAS_ERRNO_H) && VOS_HAS_ERRNO_H != 0
#include <errno.h>
#endif

#if defined(VOS_HAS_NETDB_H) && VOS_HAS_NETDB_H != 0
#include <netdb.h>
#endif

#if defined(VOS_HAS_UNISTD_H) && VOS_HAS_UNISTD_H != 0
#include <unistd.h>
#endif

#if (OS_WIN32 == 1)
#define OSERR_EWOULDBLOCK    WSAEWOULDBLOCK
#define OSERR_EINPROGRESS    WSAEINPROGRESS
#define OSERR_ECONNRESET     WSAECONNRESET
#define OSERR_ENOTCONN       WSAENOTCONN
#else
#define OSERR_EWOULDBLOCK    EWOULDBLOCK
#define OSERR_EINPROGRESS    EINPROGRESS
#define OSERR_ECONNRESET     ECONNRESET
#define OSERR_ENOTCONN       ENOTCONN
#endif

#undef s_addr
#undef s6_addr

#if !defined(VOS_HAS_SOCKLEN_T) || VOS_HAS_SOCKLEN_T==0
typedef int socklen_t;
#endif

#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
#define VOS_SOCKADDR_SET_LEN(addr,len) (((vos_addr_hdr*)(addr))->sa_zero_len=(len))
#define VOS_SOCKADDR_RESET_LEN(addr)   (((vos_addr_hdr*)(addr))->sa_zero_len=0)
#else
#define VOS_SOCKADDR_SET_LEN(addr,len) 
#define VOS_SOCKADDR_RESET_LEN(addr)
#endif

#endif	/* __VOS_COMPAT_SOCKET_H__ */


