
#ifndef __VOS_ADDR_RESOLV_H__
#define __VOS_ADDR_RESOLV_H__

#include "vos_types.h"
#include "vos_os.h"
#include "vos_sock.h"

#undef EXT
#ifndef __ADDR_RESOLV_C__
#define EXT	extern
#else
#define EXT 
#endif

#define h_addr h_addr_list[0]

typedef struct vos_hostent
{
    char                        *h_name;			/**< The official name of the host. */
    char                        **h_aliases;		/**< Aliases list. */
    int	                        h_addrtype;			/**< Host address type. */
    int	                        h_length;			/**< Length of address. */
    char                        **h_addr_list;		/**< List of addresses. */
} vos_hostent;

typedef struct vos_addrinfo
{
    char	                    ai_canonname[VOS_MAX_HOSTNAME];	 /**< Canonical name for host*/
    vos_sockaddr                ai_addr;						 /**< Binary address.	    */
} vos_addrinfo;

EXT int vos_gethostbyname(const char *hostname, vos_hostent *he);
EXT vos_status_t vos_gethostip(int af, vos_sockaddr *addr);
EXT vos_status_t vos_getaddrinfo(int af, const char *nodename, unsigned *count, vos_addrinfo ai[]);

#endif	/* __VOS_ADDR_RESOLV_H__ */


