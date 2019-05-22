
#ifndef __VOS_SOCK_H__
#define __VOS_SOCK_H__

#include "vos_types.h"

#if (OS_WIN32 == 1)
#pragma comment(lib,"ws2_32.lib")
#endif

#undef EXT
#ifndef __SOCK_C__
#define EXT    extern
#else
#define EXT
#endif


VOS_BEGIN_DECL 

#ifndef SO_NOSIGPIPE
#define SO_NOSIGPIPE 0xFFFF
#endif

/**
 * Flag to be specified in #vos_sock_shutdown().
 */
typedef enum vos_socket_sd_type
{
    VOS_SD_RECEIVE   = 0,    /**< No more receive.	    */
    VOS_SHUT_RD	    = 0,    /**< Alias for SD_RECEIVE.	    */
    VOS_SD_SEND	    = 1,    /**< No more sending.	    */
    VOS_SHUT_WR	    = 1,    /**< Alias for SD_SEND.	    */
    VOS_SD_BOTH	    = 2,    /**< No more send and receive.  */
    VOS_SHUT_RDWR    = 2     /**< Alias for SD_BOTH.	    */
} vos_socket_sd_type;


#define VOS_INADDR_ANY	    ((vos_uint32_t)0)
#define VOS_INADDR_NONE	    ((vos_uint32_t)0xffffffff)
#define VOS_INADDR_BROADCAST ((vos_uint32_t)0xffffffff)


#define VOS_INVALID_SOCKET   (-1)

#undef s_addr /* 必须undefine s_addr，下面的定义中有定义 */
typedef struct vos_in_addr
{
    vos_uint32_t	s_addr;		/*The 32bit IP address.*/
} vos_in_addr;


#define VOS_INET_ADDRSTRLEN	16
#define VOS_INET6_ADDRSTRLEN	46

/**
 * The size of sin_zero field in vos_sockaddr_in structure. Most OSes
 * use 8, but others such as the BSD TCP/IP stack in eCos uses 24.
 */
#ifndef VOS_SOCKADDR_IN_SIN_ZERO_LEN
#   define VOS_SOCKADDR_IN_SIN_ZERO_LEN	    8
#endif


struct vos_sockaddr_in
{
#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
    vos_uint8_t  sin_zero_len;
    vos_uint8_t  sin_family;
#else
    vos_uint16_t	sin_family;
#endif
    vos_uint16_t	sin_port;
    vos_in_addr	sin_addr;
    char	sin_zero[VOS_SOCKADDR_IN_SIN_ZERO_LEN]; /*Padding*/
};


#undef s6_addr
typedef union vos_in6_addr
{
    vos_uint8_t  s6_addr[16];
    vos_uint32_t	u6_addr32[4];
} vos_in6_addr;


#define VOS_IN6ADDR_ANY_INIT { { { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 } } }

#define VOS_IN6ADDR_LOOPBACK_INIT { { { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 } } }

typedef struct vos_sockaddr_in6
{
#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
    vos_uint8_t  sin6_zero_len;	    /**< Just ignore this.	   */
    vos_uint8_t  sin6_family;	    /**< Address family.	   */
#else
    vos_uint16_t	sin6_family;	    /**< Address family		    */
#endif
    vos_uint16_t	sin6_port;	    /**< Transport layer port number. */
    vos_uint32_t	sin6_flowinfo;	    /**< IPv6 flow information	    */
    vos_in6_addr sin6_addr;	    /**< IPv6 address.		    */
    vos_uint32_t sin6_scope_id;	    /**< Set of interfaces for a scope	*/
} vos_sockaddr_in6;

typedef struct vos_addr_hdr
{
#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
    vos_uint8_t  sa_zero_len;
    vos_uint8_t  sa_family;
#else
    vos_uint16_t	sa_family;
#endif
} vos_addr_hdr;

typedef union vos_sockaddr
{
    vos_addr_hdr	    addr;	    /* Generic transport address. */
    vos_sockaddr_in  ipv4;	/* IPv4 transport address.	    */
    vos_sockaddr_in6 ipv6;	/* IPv6 transport address.	    */
} vos_sockaddr;

typedef struct vos_ip_mreq 
{
    vos_in_addr imr_multiaddr;	/* IP multicast address of group. */
    vos_in_addr imr_interface;	/* local IP address of interface. */
} vos_ip_mreq;


/* *
* SOCKET ADDRESS 操作
*/
EXT vos_uint16_t			vos_ntohs(vos_uint16_t netshort);
EXT vos_uint16_t			vos_htons(vos_uint16_t hostshort);
EXT vos_uint32_t			vos_ntohl(vos_uint32_t netlong);
EXT vos_uint32_t			vos_htonl(vos_uint32_t hostlong);
EXT char*				vos_inet_ntoa(vos_in_addr inaddr);
EXT int					vos_inet_aton(const char *cp, struct vos_in_addr *inp);
EXT vos_status_t			vos_inet_pton(int af, const char *src, void *dst);
EXT vos_status_t			vos_inet_ntop(int af, const void *src, char *dst, int size);
EXT char*				vos_inet_ntop2( int af, const void *src, char *dst, int size);
EXT vos_in_addr	vos_inet_addr(const vos_str_t *cp);
EXT vos_in_addr	vos_inet_addr2(const char *cp);

EXT char*				vos_sockaddr_print( const vos_sockaddr_t *addr, char *buf, int size, unsigned flags);
EXT vos_uint16_t			vos_sockaddr_get_port(const vos_sockaddr_t *addr);
EXT unsigned			vos_sockaddr_get_addr_len(const vos_sockaddr_t *addr);
EXT unsigned			vos_sockaddr_get_len(const vos_sockaddr_t *addr);
EXT void				vos_sockaddr_cp(vos_sockaddr_t *dst, const vos_sockaddr_t *src);
EXT void				vos_sockaddr_copy_addr( vos_sockaddr *dst, const vos_sockaddr *src);
EXT void*				vos_sockaddr_get_addr(const vos_sockaddr_t *addr);
EXT void				vos_sockaddr_in_set_port(vos_sockaddr_in *addr,  vos_uint16_t hostport);
EXT vos_status_t			vos_sockaddr_set_port(vos_sockaddr *addr, vos_uint16_t hostport);
EXT vos_in_addr	vos_sockaddr_in_get_addr(const vos_sockaddr_in *addr);
EXT void				vos_sockaddr_in_set_addr(vos_sockaddr_in *addr, vos_uint32_t hostaddr);
EXT vos_status_t			vos_sockaddr_in_set_str_addr( vos_sockaddr_in *addr, const vos_str_t *str_addr);
EXT vos_status_t			vos_sockaddr_set_str_addr(int af, vos_sockaddr *addr, const vos_str_t *str_addr);
EXT vos_status_t			vos_sockaddr_in_init( vos_sockaddr_in *addr, const vos_str_t *str_addr, vos_uint16_t port);
EXT vos_status_t			vos_sockaddr_init(int af, vos_sockaddr *addr, const vos_str_t *cp, vos_uint16_t port);
EXT int					vos_sockaddr_cmp( const vos_sockaddr_t *addr1, const vos_sockaddr_t *addr2);
EXT vos_in_addr	vos_gethostaddr(void);

EXT vos_status_t			vos_getipinterface(int af, const vos_str_t *dst, vos_sockaddr *itf_addr, vos_bool_t allow_resolve, vos_sockaddr *p_dst_addr);
EXT vos_status_t			vos_getdefaultipinterface(int af, vos_sockaddr *addr);
EXT const vos_str_t*		vos_gethostname(void);

/**
 * SOCKET API.
 */
EXT vos_status_t vos_sock_socket(int family, 
						int type, 
						int protocol, 
						vos_sock_t *sock);
EXT vos_status_t vos_sock_close(vos_sock_t sockfd);
EXT vos_status_t vos_sock_bind( vos_sock_t sockfd, 
						const vos_sockaddr_t *my_addr, 
						int addrlen);
EXT vos_status_t vos_sock_bind_in(vos_sock_t sockfd, 
						vos_uint32_t addr, 
						vos_uint16_t port);

EXT vos_status_t vos_sock_listen( vos_sock_t sockfd, int backlog );
EXT vos_status_t vos_sock_accept( vos_sock_t serverfd, 
						vos_sock_t *newsock, 
						vos_sockaddr_t *addr, 
						int *addrlen);

EXT vos_status_t vos_sock_connect( vos_sock_t sockfd, 
					const vos_sockaddr_t *serv_addr, 
					int addrlen);
EXT vos_status_t vos_sock_getpeername(vos_sock_t sockfd, 
					vos_sockaddr_t *addr, 
					int *namelen);
EXT vos_status_t vos_sock_getsockname( vos_sock_t sockfd, 
					vos_sockaddr_t *addr, 
					int *namelen);
EXT vos_status_t vos_sock_getsockopt( vos_sock_t sockfd,
					 vos_uint16_t level,
					 vos_uint16_t optname,
					 void *optval,
					 int *optlen);
EXT vos_status_t vos_sock_setsockopt( vos_sock_t sockfd,
					 vos_uint16_t level,
					 vos_uint16_t optname,
					 const void *optval,
					 int optlen);
EXT vos_status_t vos_sock_recv(vos_sock_t sockfd,
				  void *buf,
				  vos_ssize_t *len,
				  unsigned flags);
EXT vos_status_t vos_sock_recvfrom( vos_sock_t sockfd,
				      void *buf,
				      vos_ssize_t *len,
				      unsigned flags,
				      vos_sockaddr_t *from,
				      int *fromlen);
EXT vos_status_t vos_sock_send(vos_sock_t sockfd,
				  const void *buf,
				  vos_ssize_t *len,
				  unsigned flags);
EXT vos_status_t vos_sock_sendto(vos_sock_t sockfd,
   				    const void *buf,
   				    vos_ssize_t *len,
   				    unsigned flags,
   				    const vos_sockaddr_t *to,
   				    int tolen);

EXT vos_status_t vos_sock_shutdown( vos_sock_t sockfd, int how);

VOS_END_DECL

#endif	/* __VOS_SOCK_H__ */


