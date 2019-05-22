#define __SOCK_C__

#include "vos_sock.h"
#include "vos_os.h"
#include "vos_assert.h"
#include "vos_string.h"
#include "vos_socket.h"
#include "vos_addr_resolv.h"
#include "vos_unicode.h"
#include "vos_log.h"

vos_status_t vos_get_netos_error(void)
{
#if  (OS_WIN32 == 1)
    return WSAGetLastError();
#else
    return errno;
#endif
}

#if 0
static void CHECK_ADDR_LEN(const vos_sockaddr *addr, int len)
{
    vos_sockaddr *a = (vos_sockaddr*)addr;
    vos_assert((a->addr.sa_family==AF_INET && len==sizeof(vos_sockaddr_in)) ||
	      (a->addr.sa_family==AF_INET6 && len==sizeof(vos_sockaddr_in6)));

}
#else
#define CHECK_ADDR_LEN(addr,len)
#endif

vos_uint16_t vos_ntohs(vos_uint16_t netshort)
{
    return ntohs(netshort);
}

vos_uint16_t vos_htons(vos_uint16_t hostshort)
{
    return htons(hostshort);
}

vos_uint32_t vos_ntohl(vos_uint32_t netlong)
{
    return ntohl(netlong);
}

vos_uint32_t vos_htonl(vos_uint32_t hostlong)
{
    return htonl(hostlong);
}

char* vos_inet_ntoa(vos_in_addr inaddr)
{
#if (OS_LINUX == 0)
    return inet_ntoa(*(struct in_addr*)&inaddr);
#else
    struct in_addr addr;
    addr.s_addr = inaddr.s_addr;
    return inet_ntoa(addr);
#endif
}

int vos_inet_aton(const char *cp, struct vos_in_addr *inp)
{
    inp->s_addr = VOS_INADDR_NONE;

    /* Caution:
     *	this function might be called with cp->slen >= 16
     *  (i.e. when called with hostname to check if it's an IP addr).
     */
    VOS_ASSERT_RETURN(cp && strlen(cp) && inp, 0);
    if (strlen(cp) >= VOS_INET_ADDRSTRLEN) 
    {
	    return 0;
    }

#if defined(VOS_SOCK_HAS_INET_ATON) && VOS_SOCK_HAS_INET_ATON != 0
    return inet_aton(cp, (struct in_addr*)inp);
#else
    inp->s_addr = inet_addr(cp);
    return inp->s_addr == VOS_INADDR_NONE ? 0 : 1;
#endif
}

/*
 * Convert text to IPv4/IPv6 address.
 */

vos_status_t vos_inet_pton(int af, const char *src, void *dst)
{
#if 1
#define VOS_EAFNOTSUP -1
    VOS_ASSERT_RETURN(af==AF_INET || af==AF_INET6, VOS_EAFNOTSUP);
    VOS_ASSERT_RETURN(src && strlen(src) && dst, VOS_EINVAL);

    /* Initialize output with VOS_IN_ADDR_NONE for IPv4 (to be 
     * compatible with vos_inet_aton()
     */
    if (af==AF_INET) 
    {
	    ((vos_in_addr*)dst)->s_addr = VOS_INADDR_NONE;
    }

    /* Caution:
     *	this function might be called with cp->slen >= 46
     *  (i.e. when called with hostname to check if it's an IP addr).
     */
    if (strlen(src) >= VOS_INET6_ADDRSTRLEN) 
    {
	    return VOS_ENAMETOOLONG;
    }

#if (OS_LINUX == 1)
    if (inet_pton(af, src, dst) != 1) 
    {
    	vos_status_t status = vos_get_netos_error();
    	if (status == VOS_SUCCESS)
    	{
    	    status = VOS_EUNKNOWN;
    	}
    	return status;
    }

    return VOS_SUCCESS;

#elif (OS_WIN32 == 1)
    {
    	VOS_DECL_UNICODE_TEMP_BUF(wtempaddr,VOS_INET6_ADDRSTRLEN)
    	vos_sockaddr sock_addr;
    	int addr_len = sizeof(sock_addr);
    	int rc;

    	sock_addr.addr.sa_family = (vos_uint16_t)af;
		rc = WSAStringToAddress(
    				VOS_STRING_TO_NATIVE(src,wtempaddr,sizeof(wtempaddr)), 
    				af, NULL, (LPSOCKADDR)&sock_addr, &addr_len);
    	if (rc != 0) 
    	{
    	    vos_status_t status = vos_get_netos_error();
    	    if (status == VOS_SUCCESS)
    		{
    		    status = -1;
            }
    	    return status;
    	}

    	if (sock_addr.addr.sa_family == AF_INET) 
    	{
    	    memcpy(dst, &sock_addr.ipv4.sin_addr, 4);
    	    return VOS_SUCCESS;
    	} 
    	else if (sock_addr.addr.sa_family == AF_INET6) 
    	{
    	    memcpy(dst, &sock_addr.ipv6.sin6_addr, 16);
    	    return VOS_SUCCESS;
    	}
    	else 
    	{
    	    vos_assert(!"Shouldn't happen");
    	    return -1;
    	}
    }
#elif !defined(VOS_HAS_IPV6) || VOS_HAS_IPV6==0
    return VOS_EIPV6NOTSUP;
#else
    vos_assert(!"Not supported");
    return VOS_EIPV6NOTSUP;
#endif
#endif
    return -1;
}

vos_status_t vos_inet_ntop(int af, const void *src,
				 char *dst, int size)
{
    VOS_ASSERT_RETURN(src && dst && size, VOS_EINVAL);

    *dst = '\0';

    VOS_ASSERT_RETURN(af==AF_INET || af==AF_INET6, -1);

#if (OS_LINUX == 1)
    if (inet_ntop(af, src, dst, size) == NULL) 
    {
    	vos_status_t status = vos_get_netos_error();
    	if (status == VOS_SUCCESS)
    	{
    	    status = VOS_EUNKNOWN;
        }
    	return status;
    }

    return VOS_SUCCESS;

#elif (OS_WIN32 == 1)
    /*
     * Implementation on Windows, using WSAAddressToString().
     * Should also work on Unicode systems.
     */
    {
	VOS_DECL_UNICODE_TEMP_BUF(wtempaddr,VOS_INET6_ADDRSTRLEN)
	vos_sockaddr sock_addr;
	DWORD addr_len, addr_str_len;
	int rc;

	vos_bzero(&sock_addr, sizeof(sock_addr));
	sock_addr.addr.sa_family = (vos_uint16_t)af;
	if (af == AF_INET) 
	{
	    if (size < VOS_INET_ADDRSTRLEN)
		return VOS_ETOOSMALL;
	    memcpy(&sock_addr.ipv4.sin_addr, src, 4);
	    addr_len = sizeof(vos_sockaddr_in);
	    addr_str_len = VOS_INET_ADDRSTRLEN;
	} 
	else if (af == AF_INET6) 
	{
	    if (size < VOS_INET6_ADDRSTRLEN)
		return VOS_ETOOSMALL;
	    memcpy(&sock_addr.ipv6.sin6_addr, src, 16);
	    addr_len = sizeof(vos_sockaddr_in6);
	    addr_str_len = VOS_INET6_ADDRSTRLEN;
	} 
	else 
	{
	    vos_assert(!"Unsupported address family");
        return -1;// VOS_EAFNOTSUP;
	}

#if _UNICODE
	rc = WSAAddressToString((LPSOCKADDR)&sock_addr, addr_len,
				NULL, wtempaddr, &addr_str_len);
	if (rc == 0) 
	{
	    vos_unicode_to_ansi(wtempaddr, wcslen(wtempaddr), dst, size);
	}
#else
	rc = WSAAddressToString((LPSOCKADDR)&sock_addr, addr_len,
				NULL, dst, &addr_str_len);
#endif

	if (rc != 0) {
	    vos_status_t status = vos_get_netos_error();
	    if (status == VOS_SUCCESS)
		status = -1;

	    return status;
	}

	return VOS_SUCCESS;
    }

#elif !defined(VOS_HAS_IPV6) || VOS_HAS_IPV6==0
    /* IPv6 support is disabled, just return error without raising assertion */
    return VOS_EIPV6NOTSUP;
#else
    vos_assert(!"Not supported");
    return VOS_EIPV6NOTSUP;
#endif
}

char* vos_inet_ntop2( int af, const void *src, char *dst, int size)
{
	vos_status_t status;

	status = vos_inet_ntop(af, src, dst, size);
	return (status==VOS_SUCCESS)? dst : NULL;
}

/*
 * Print socket address.
 */
char* vos_sockaddr_print( const vos_sockaddr_t *addr,
				 char *buf, int size,
				 unsigned flags)
{
    enum 
	{
		WITH_PORT = 1,
		WITH_BRACKETS = 2
    };

    char txt[VOS_INET6_ADDRSTRLEN];
    char port[32];
    const vos_addr_hdr *h = (const vos_addr_hdr*)addr;
    char *bquote, *equote;
    vos_status_t status;

    status = vos_inet_ntop(h->sa_family, vos_sockaddr_get_addr(addr),
			  txt, sizeof(txt));
    if (status != VOS_SUCCESS)
	return "";

    if (h->sa_family != AF_INET6 || (flags & WITH_BRACKETS)==0) 
	{
		bquote = ""; equote = "";
    } 
	else 
	{
		bquote = "["; equote = "]";
    }

    if (flags & WITH_PORT) 
	{
		vos_ansi_snprintf(port, sizeof(port), ":%d", vos_sockaddr_get_port(addr));
    } 
	else 
	{
		port[0] = '\0';
    }

    vos_ansi_snprintf(buf, size, "%s%s%s%s", bquote, txt, equote, port);

    return buf;
}

/*
 * Get port number
 */
vos_uint16_t vos_sockaddr_get_port(const vos_sockaddr_t *addr)
{
    const vos_sockaddr *a = (const vos_sockaddr*) addr;

    VOS_ASSERT_RETURN(a->addr.sa_family == AF_INET ||
		     a->addr.sa_family == AF_INET6, (vos_uint16_t)0xFFFF);

    return vos_ntohs((vos_uint16_t)(a->addr.sa_family == AF_INET6 ?
				    a->ipv6.sin6_port : a->ipv4.sin_port));
}

/*
 * Get the length of the address part.
 */
unsigned vos_sockaddr_get_addr_len(const vos_sockaddr_t *addr)
{
    const vos_sockaddr *a = (const vos_sockaddr*) addr;
    VOS_ASSERT_RETURN(a->addr.sa_family == AF_INET ||
		     a->addr.sa_family == AF_INET6, 0);
    return a->addr.sa_family == AF_INET6 ?
	    sizeof(vos_in6_addr) : sizeof(vos_in_addr);
}


unsigned vos_sockaddr_get_len(const vos_sockaddr_t *addr)
{
	const vos_sockaddr *a = (const vos_sockaddr*) addr;
	VOS_ASSERT_RETURN(a->addr.sa_family == AF_INET ||
		a->addr.sa_family == AF_INET6, 0);
	return a->addr.sa_family == AF_INET6 ?
		sizeof(vos_sockaddr_in6) : sizeof(vos_sockaddr_in);
}

/*
 * Copy only the address part (sin_addr/sin6_addr) of a socket address.
 */
void vos_sockaddr_copy_addr( vos_sockaddr *dst, const vos_sockaddr *src)
{
    /* Destination sockaddr might not be initialized */
    const char *srcbuf = (char*)vos_sockaddr_get_addr(src);
    char *dstbuf = ((char*)dst) + (srcbuf - (char*)src);
    memcpy(dstbuf, srcbuf, vos_sockaddr_get_addr_len(src));
}

/*
 * Copy socket address.
 */
void vos_sockaddr_cp(vos_sockaddr_t *dst, const vos_sockaddr_t *src)
{
    memcpy(dst, src, vos_sockaddr_get_len(src));
}

/*
 * Get the address part
 */
void* vos_sockaddr_get_addr(const vos_sockaddr_t *addr)
{
    const vos_sockaddr *a = (const vos_sockaddr*)addr;

    VOS_ASSERT_RETURN(a->addr.sa_family == AF_INET ||
		     a->addr.sa_family == AF_INET6, NULL);

    if (a->addr.sa_family == AF_INET6)
	return (void*) &a->ipv6.sin6_addr;
    else
	return (void*) &a->ipv4.sin_addr;
}

/*
 * Set port number of vos_sockaddr_in
 */
void vos_sockaddr_in_set_port(vos_sockaddr_in *addr,  vos_uint16_t hostport)
{
    addr->sin_port = vos_htons(hostport);
}

/*
 * Set port number of vos_sockaddr
 */
vos_status_t vos_sockaddr_set_port(vos_sockaddr *addr, vos_uint16_t hostport)
{
    int af = addr->addr.sa_family;

    VOS_ASSERT_RETURN(af==AF_INET || af==AF_INET6, VOS_EINVAL);

    if (af == AF_INET6)
	addr->ipv6.sin6_port = vos_htons(hostport);
    else
	addr->ipv4.sin_port = vos_htons(hostport);

    return VOS_SUCCESS;
}

/*
 * Get IPv4 address
 */
vos_in_addr vos_sockaddr_in_get_addr(const vos_sockaddr_in *addr)
{
    vos_in_addr in_addr;
    in_addr.s_addr = vos_ntohl(addr->sin_addr.s_addr);
    return in_addr;
}

/*
 * Set IPv4 address
 */
void vos_sockaddr_in_set_addr(vos_sockaddr_in *addr,
				     vos_uint32_t hostaddr)
{
    addr->sin_addr.s_addr = vos_htonl(hostaddr);
}

/*
 * Convert address string with numbers and dots to binary IP address.
 */ 
vos_in_addr vos_inet_addr(const vos_str_t *cp)
{
	vos_in_addr addr;
	vos_char_t szcp[32];

	vos_inet_aton(vos_str2(cp, szcp, sizeof(szcp)), &addr);

	return addr;
}

/*
 * Convert address string with numbers and dots to binary IP address.
 */ 
vos_in_addr vos_inet_addr2(const char *cp)
{
    vos_str_t str = vos_str((char*)cp);
    return vos_inet_addr(&str);
}


vos_status_t vos_sockaddr_in_set_str_addr( vos_sockaddr_in *addr, const vos_str_t *str_addr)
{

	VOS_ASSERT_RETURN(!str_addr || str_addr->slen < VOS_MAX_HOSTNAME, 
		(addr->sin_addr.s_addr=VOS_INADDR_NONE, VOS_EINVAL));

	VOS_SOCKADDR_RESET_LEN(addr);

	addr->sin_family = AF_INET;
	//vos_bzero(addr->sin_zero, sizeof(addr->sin_zero));
    //memset(addr->sin_zero, 0x0, sizeof(addr->sin_zero));
    
	if (str_addr && str_addr->slen) 
	{
		addr->sin_addr = vos_inet_addr(str_addr);
		if (addr->sin_addr.s_addr == VOS_INADDR_NONE) 
		{
			vos_hostent he;
			vos_status_t rc;
			char straddr[32];

			rc = vos_gethostbyname(vos_str2(str_addr, straddr, sizeof(straddr)), &he);
			if (rc == 0) 
			{
				addr->sin_addr.s_addr = *(vos_uint32_t*)he.h_addr;
			} 
			else 
			{
				addr->sin_addr.s_addr = VOS_INADDR_NONE;
				return rc;
			}
		}

	} else {
		addr->sin_addr.s_addr = 0;
	}

	return VOS_SUCCESS;
}


vos_status_t vos_sockaddr_set_str_addr(int af,
	vos_sockaddr *addr,
	const vos_str_t *str_addr)
{
	vos_status_t status;
	char straddr[32];
	vos_addrinfo ai;
	unsigned count = 1;

	if (af == AF_INET) 
	{
		return vos_sockaddr_in_set_str_addr(&addr->ipv4, str_addr);
	}

	VOS_ASSERT_RETURN(af==AF_INET6, -1);

	vos_str2(str_addr, straddr, sizeof(straddr));

	/* IPv6 specific */

	addr->ipv6.sin6_family = AF_INET6;
	VOS_SOCKADDR_RESET_LEN(addr);

	if (str_addr && str_addr->slen) 
	{
		status = vos_inet_pton(AF_INET6, straddr, &addr->ipv6.sin6_addr);
		if (status != VOS_SUCCESS) 
		{


			status = vos_getaddrinfo(AF_INET6, straddr, &count, &ai);
			if (status==VOS_SUCCESS) 
			{
				memcpy(&addr->ipv6.sin6_addr, &ai.ai_addr.ipv6.sin6_addr,
					sizeof(vos_sockaddr_in6));
			}
		}
	} 
	else 
	{
		status = VOS_SUCCESS;
	}

	return status;
}


/*
 * Set the IP address and port of an IP socket address.
 * The string address may be in a standard numbers and dots notation or 
 * may be a hostname. If hostname is specified, then the function will 
 * resolve the host into the IP address.
 */
vos_status_t vos_sockaddr_in_init( vos_sockaddr_in *addr,
	const vos_str_t *str_addr,
	vos_uint16_t port)
{
	vos_assert(addr && str_addr);

	addr->sin_family = AF_INET;
	vos_sockaddr_in_set_port(addr, port);
	return vos_sockaddr_in_set_str_addr(addr, str_addr);
}

/*
 * Initialize IP socket address based on the address and port info.
 */
vos_status_t vos_sockaddr_init(int af, 
				     vos_sockaddr *addr,
				     const vos_str_t *cp,
				     vos_uint16_t port)
{
    vos_status_t status;

    if (af == AF_INET) 
	{
		return vos_sockaddr_in_init(&addr->ipv4, cp, port);
    }

    /* IPv6 specific */
    VOS_ASSERT_RETURN(af==AF_INET6, -1);

    vos_bzero(addr, sizeof(vos_sockaddr_in6));
    addr->addr.sa_family = AF_INET6;
    
    status = vos_sockaddr_set_str_addr(af, addr, cp);
    if (status != VOS_SUCCESS)
	return status;

    addr->ipv6.sin6_port = vos_htons(port);
    return VOS_SUCCESS;
}


/*
 * Compare two socket addresses.
 */
int vos_sockaddr_cmp( const vos_sockaddr_t *addr1, const vos_sockaddr_t *addr2)
{
    const vos_sockaddr *a1 = (const vos_sockaddr*) addr1;
    const vos_sockaddr *a2 = (const vos_sockaddr*) addr2;
    int port1, port2;
    int result;

    /* Compare address family */
    if (a1->addr.sa_family < a2->addr.sa_family)
	{
		return -1;
	}
    else if (a1->addr.sa_family > a2->addr.sa_family)
	{
		return 1;
	}

    /* Compare addresses */
    result = memcmp(vos_sockaddr_get_addr(a1),
									vos_sockaddr_get_addr(a2),
									vos_sockaddr_get_addr_len(a1));
    if (result != 0)
	{
		return result;
	}
	

    /* Compare port number */
    port1 = vos_sockaddr_get_port(a1);
    port2 = vos_sockaddr_get_port(a2);

    if (port1 < port2)
	{
		return -1;
	}
	else if (port1 > port2)
	{
		return 1;
	}
    /* TODO:
     *	Do we need to compare flow label and scope id in IPv6? 
     */
    
    /* Looks equal */
    return 0;
}

/*
 * Get first IP address associated with the hostname.
 */
vos_in_addr vos_gethostaddr(void)
{
    vos_sockaddr_in addr;
    const vos_str_t *hostname = vos_gethostname();

    vos_sockaddr_in_set_str_addr(&addr, hostname);
    return addr.sin_addr;
}


/* Get IP interface for sending to the specified destination */
vos_status_t vos_getipinterface(int af,
	const vos_str_t *dst,
	vos_sockaddr *itf_addr,
	vos_bool_t allow_resolve,
	vos_sockaddr *p_dst_addr)
{
	vos_sockaddr dst_addr;
	vos_sock_t fd;
	int len;
	vos_uint8_t zero[64];
	vos_status_t status;
	char strdst[32];

	vos_str2(dst, strdst, sizeof(strdst));

	vos_sockaddr_init(af, &dst_addr, dst, 53);
	status = vos_inet_pton(af, strdst, vos_sockaddr_get_addr(&dst_addr));
	if (status != VOS_SUCCESS) 
	{
		/* "dst" is not an IP address. */
		if (allow_resolve) 
		{
			status = vos_sockaddr_init(af, &dst_addr, dst, 53);
		} 
		else 
		{
			vos_str_t cp;

			if (af == AF_INET) 
			{
				cp = vos_str("1.1.1.1");
			} 
			else 
			{
				cp = vos_str("1::1");
			}
			status = vos_sockaddr_init(af, &dst_addr, &cp, 53);
		}

		if (status != VOS_SUCCESS)
			return status;
	}

	/* Create UDP socket and connect() to the destination IP */
	status = vos_sock_socket(af, SOCK_DGRAM, 0, &fd);
	if (status != VOS_SUCCESS) 
	{
		return status;
	}

	status = vos_sock_connect(fd, &dst_addr, vos_sockaddr_get_len(&dst_addr));
	if (status != VOS_SUCCESS) 
	{
		vos_sock_close(fd);
		return status;
	}

	len = sizeof(*itf_addr);
	status = vos_sock_getsockname(fd, itf_addr, &len);
	if (status != VOS_SUCCESS) 
	{
		vos_sock_close(fd);
		return status;
	}

	vos_sock_close(fd);

	/* Check that the address returned is not zero */
	vos_bzero(zero, sizeof(zero));
	if (memcmp(vos_sockaddr_get_addr(itf_addr), zero,
		vos_sockaddr_get_addr_len(itf_addr))==0)
	{
		return -1;
	}

	if (p_dst_addr)
		*p_dst_addr = dst_addr;

	return VOS_SUCCESS;
}

/* Get the default IP interface */
vos_status_t vos_getdefaultipinterface(int af, vos_sockaddr *addr)
{
	vos_str_t cp;

	if (af == AF_INET) 
	{
		cp = vos_str("1.1.1.1");
	} 
	else 
	{
		cp = vos_str("1::1");
	}

	return vos_getipinterface(af, &cp, addr, FALSE, NULL);
}


/*
 * Get hostname.
 */
const vos_str_t* vos_gethostname(void)
{
    static char buf[VOS_MAX_HOSTNAME];
    static vos_str_t hostname;

    if (hostname.ptr == NULL) 
    {
    	hostname.ptr = buf;
    	if (gethostname(buf, sizeof(buf)) != 0) 
    	{
    	    hostname.ptr[0] = '\0';
    	    hostname.slen = 0;
    	} 
    	else 
    	{
                hostname.slen = strlen(buf);
    	}
    }
    return &hostname;
}

vos_status_t vos_sock_socket(int af, 
    int type, 
    int proto,
    vos_sock_t *sock)
{
#if (OS_WIN32 == 1)
    VOS_ASSERT_RETURN(sock!=NULL, VOS_EINVAL);
    VOS_ASSERT_RETURN((unsigned)VOS_INVALID_SOCKET==INVALID_SOCKET, 
        (*sock=VOS_INVALID_SOCKET, VOS_EINVAL));

    *sock = WSASocket(af, type, proto, NULL, 0, WSA_FLAG_OVERLAPPED);

    if (*sock == VOS_INVALID_SOCKET) 
    {
        return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
#if VOS_SOCK_DISABLE_WSAECONNRESET

#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR,12)
#endif
    /* Disable WSAECONNRESET for UDP.
    */
    if (type==SOCK_DGRAM) 
    {
        DWORD dwBytesReturned = 0;
        BOOL bNewBehavior = FALSE;
        DWORD rc;

        rc = WSAIoctl(*sock, SIO_UDP_CONNRESET,
            &bNewBehavior, sizeof(bNewBehavior),
            NULL, 0, &dwBytesReturned,
            NULL, NULL);

        if (rc==SOCKET_ERROR) 
        {
            // Ignored..
        }
    }
#endif
    return VOS_SUCCESS;

#else
    VOS_ASSERT_RETURN(sock!=NULL, VOS_EINVAL);
    VOS_ASSERT_RETURN(VOS_INVALID_SOCKET==-1, 
        (*sock=VOS_INVALID_SOCKET, VOS_EINVAL));

    *sock = socket(af, type, proto);
    if (*sock == VOS_INVALID_SOCKET)
    {
        return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else 
    {
        vos_int32_t val = 1;
        if (type == SOCK_STREAM) 
        {
            vos_sock_setsockopt(*sock, SOL_SOCKET, SO_NOSIGPIPE,
                &val, sizeof(val));
        }

        return VOS_SUCCESS;
    }
#endif
}


vos_status_t vos_sock_bind( vos_sock_t sock, 
				  const vos_sockaddr_t *addr,
				  int len)
{
    VOS_ASSERT_RETURN(addr && len >= (int)sizeof(struct sockaddr_in), VOS_EINVAL);
    CHECK_ADDR_LEN(addr, len);
    if (bind(sock, (struct sockaddr*)addr, len) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else
    {
	    return VOS_SUCCESS;
    }
}

vos_status_t vos_sock_bind_in( vos_sock_t sock, 
				     vos_uint32_t addr32,
				     vos_uint16_t port)
{
    vos_sockaddr_in addr;

    VOS_SOCKADDR_SET_LEN(&addr, sizeof(vos_sockaddr_in));
    vos_bzero(&addr, sizeof(addr));
    addr.sin_family = AF_INET;
    //vos_bzero(addr.sin_zero, sizeof(addr.sin_zero));
    addr.sin_addr.s_addr = vos_htonl(addr32);
    addr.sin_port = vos_htons(port);
    
    return vos_sock_bind(sock, &addr, sizeof(vos_sockaddr_in));
}

vos_status_t vos_sock_close(vos_sock_t sock)
{
    int rc;
    
#if (OS_WIN32 == 1)
    rc = closesocket(sock);
#else
    rc = close(sock);
#endif

    if (rc != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else
    {
	    return VOS_SUCCESS;
	}
}

vos_status_t vos_sock_getpeername( vos_sock_t sock,
					 vos_sockaddr_t *addr,
					 int *namelen)
{
//    //VOS_CHECK_STACK;
    if (getpeername(sock, (struct sockaddr*)addr, (vos_socklen_t*)namelen) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else 
    {
    	VOS_SOCKADDR_RESET_LEN(addr);
    	return VOS_SUCCESS;
    }
}

vos_status_t vos_sock_getsockname( vos_sock_t sock,
					 vos_sockaddr_t *addr,
					 int *namelen)
{
//    //VOS_CHECK_STACK;
    if (getsockname(sock, (struct sockaddr*)addr, (vos_socklen_t*)namelen) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else 
    {
    	VOS_SOCKADDR_RESET_LEN(addr);
    	return VOS_SUCCESS;
    }
}


vos_status_t vos_sock_send(vos_sock_t sock,
				 const void *buf,
				 vos_ssize_t *len,
				 unsigned flags)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(len, VOS_EINVAL);

    #ifdef MSG_NOSIGNAL
    /* Suppress SIGPIPE*/
    flags |= MSG_NOSIGNAL;
    #endif

    *len = send(sock, (const char*)buf, *len, flags);

	if (*len < 0)
	{
		return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
	}
    
    return VOS_SUCCESS;
}

vos_status_t vos_sock_sendto(vos_sock_t sock,
				   const void *buf,
				   vos_ssize_t *len,
				   unsigned flags,
				   const vos_sockaddr_t *to,
				   int tolen)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(len, VOS_EINVAL);
    
    CHECK_ADDR_LEN(to, tolen);

    *len = sendto(sock, (const char*)buf, *len, flags, 
		  (const struct sockaddr*)to, tolen);

	if (*len < 0) 
	{
        return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
	}

	return VOS_SUCCESS;
}

vos_status_t vos_sock_recv(vos_sock_t sock,
				 void *buf,
				 vos_ssize_t *len,
				 unsigned flags)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(buf && len, VOS_EINVAL);

    *len = recv(sock, (char*)buf, *len, flags);

    if (*len < 0) 
    {
        return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    
	return VOS_SUCCESS;
}

vos_status_t vos_sock_recvfrom(vos_sock_t sock,
				     void *buf,
				     vos_ssize_t *len,
				     unsigned flags,
				     vos_sockaddr_t *from,
				     int *fromlen)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(buf && len, VOS_EINVAL);
    VOS_ASSERT_RETURN(from && fromlen, (*len=-1, VOS_EINVAL));

    *len = recvfrom(sock, (char*)buf, *len, flags, 
		    (struct sockaddr*)from, (vos_socklen_t*)fromlen);

    if (*len < 0) 
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }

    VOS_SOCKADDR_RESET_LEN(from);
    return VOS_SUCCESS;
 }

vos_status_t vos_sock_getsockopt( vos_sock_t sock,
					vos_uint16_t level,
					vos_uint16_t optname,
					void *optval,
					int *optlen)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(optval && optlen, VOS_EINVAL);

    if (getsockopt(sock, level, optname, (char*)optval, (vos_socklen_t*)optlen)!=0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
 
	return VOS_SUCCESS;
}

vos_status_t vos_sock_setsockopt( vos_sock_t sock,
					vos_uint16_t level,
					vos_uint16_t optname,
					const void *optval,
					int optlen)
{
//    //VOS_CHECK_STACK;
    if (setsockopt(sock, level, optname, (const char*)optval, optlen) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    
	return VOS_SUCCESS;
}

vos_status_t vos_sock_connect( vos_sock_t sock,
				     const vos_sockaddr_t *addr,
				     int namelen)
{
//    //VOS_CHECK_STACK;
    if (connect(sock, (struct sockaddr*)addr, namelen) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    
	return VOS_SUCCESS;
}

#if VOS_HAS_TCP
vos_status_t vos_sock_shutdown( vos_sock_t sock,
				      int how)
{
//    //VOS_CHECK_STACK;
    if (shutdown(sock, how) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    
	return VOS_SUCCESS;
}
vos_status_t vos_sock_listen( vos_sock_t sock,
				    int backlog)
{
//    //VOS_CHECK_STACK;
    if (listen(sock, backlog) != 0)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    
	return VOS_SUCCESS;
}

vos_status_t vos_sock_accept( vos_sock_t serverfd,
				    vos_sock_t *newsock,
				    vos_sockaddr_t *addr,
				    int *addrlen)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(newsock != NULL, VOS_EINVAL);

#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
    if (addr) 
    {
	    VOS_SOCKADDR_SET_LEN(addr, *addrlen);
    }
#endif
    
    *newsock = accept(serverfd, (struct sockaddr*)addr, (vos_socklen_t*)addrlen);
    if (*newsock==VOS_INVALID_SOCKET)
    {
	    return VOS_RETURN_OS_ERROR(vos_get_native_netos_error());
    }
    else 
    {
	
#if defined(VOS_SOCKADDR_HAS_LEN) && VOS_SOCKADDR_HAS_LEN!=0
	if (addr) 
	{
	    VOS_SOCKADDR_RESET_LEN(addr);
	}
#endif
	    
	return VOS_SUCCESS;
    }
}
#endif	/* VOS_HAS_TCP */



