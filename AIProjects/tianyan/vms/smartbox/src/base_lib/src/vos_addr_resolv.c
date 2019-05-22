#define __ADDR_RESOLV_C__

#include "vos_addr_resolv.h"
#include "vos_socket.h"
#include "vos_assert.h"
#include "vos_string.h"
#include "vos_log.h"

#define THIS_FILE "addr_resolv.c"

static vos_status_t if_enum_by_af(int af,
                                  unsigned *p_cnt,
                                  vos_sockaddr ifs[])
{
    vos_status_t status;

    VOS_ASSERT_RETURN(p_cnt && *p_cnt > 0 && ifs, -1);

    vos_bzero(ifs, sizeof(ifs[0]) * (*p_cnt));

    /* Just get one default route */
    status = vos_getdefaultipinterface(af, &ifs[0]);
    if (status != VOS_SUCCESS)
        return status;

    *p_cnt = 1;
    return VOS_SUCCESS;
}

vos_status_t vos_enum_ip_interface(int af,
                                   unsigned *p_cnt,
                                   vos_sockaddr ifs[])
{
    unsigned start;
    vos_status_t status;

    start = 0;
    if (af == AF_INET6 || af == AF_UNSPEC)
    {
        unsigned max = *p_cnt;
        status = if_enum_by_af(AF_INET6, &max, &ifs[start]);
        if (status == VOS_SUCCESS)
        {
            start += max;
            (*p_cnt) -= max;
        }
    }

    if (af == AF_INET || af == AF_UNSPEC)
    {
        unsigned max = *p_cnt;
        status = if_enum_by_af(AF_INET, &max, &ifs[start]);
        if (status == VOS_SUCCESS)
        {
            start += max;
            (*p_cnt) -= max;
        }
    }

    *p_cnt = start;

    return (*p_cnt != 0) ? VOS_SUCCESS : -1;
}
///////////////////////////////////////////////////////////////////


vos_status_t vos_gethostbyname(const char *hostname, vos_hostent *phe)
{
    struct hostent *he;

    if (!hostname)
    {
        return -1;
    }

    if (strlen(hostname) >= VOS_MAX_HOSTNAME)
    {
        return VOS_ENAMETOOLONG;
    }

    he = gethostbyname(hostname);
    if (!he)
    {
        return VOS_ERESOLVE;
    }

    phe->h_name = he->h_name;
    phe->h_aliases = he->h_aliases;
    phe->h_addrtype = he->h_addrtype;
    phe->h_length = he->h_length;
    phe->h_addr_list = he->h_addr_list;

    return VOS_SUCCESS;
}

vos_status_t vos_getaddrinfo(int af, const char *nodename, unsigned *count, vos_addrinfo ai[])
{
#if defined(VOS_SOCK_HAS_GETADDRINFO) && VOS_SOCK_HAS_GETADDRINFO!=0
    char nodecopy[VOS_MAX_HOSTNAME];
    vos_bool_t has_addr = FALSE;
    unsigned i;
    int rc;
    struct addrinfo hint, *res, *orig_res;

    VOS_ASSERT_RETURN(nodename && strlen(nodename) && count && *count && ai, VOS_EINVAL);
    VOS_ASSERT_RETURN(af == AF_INET || af == AF_INET6 || af == AF_UNSPEC, VOS_EINVAL);

    /* Check if nodename is IP address */
    vos_bzero(&ai[0], sizeof(ai[0]));
    if ((af == AF_INET || af == AF_UNSPEC) &&
        vos_inet_pton(AF_INET, nodename, &ai[0].ai_addr.ipv4.sin_addr) == VOS_SUCCESS)
    {
        af = AF_INET;
        has_addr = TRUE;
    }
    else if ((af == AF_INET6 || af == AF_UNSPEC) &&
             vos_inet_pton(AF_INET6, nodename, &ai[0].ai_addr.ipv6.sin6_addr) == VOS_SUCCESS)
    {
        af = AF_INET6;
        has_addr = TRUE;
    }

    if (has_addr)
    {
        vos_str_t tmp, name;

        name = pj_str((char*)nodename);
        tmp.ptr = ai[0].ai_canonname;

        vos_strncpy_with_null(&tmp, &name, VOS_MAX_HOSTNAME);
        ai[0].ai_addr.addr.sa_family = (vos_uint16_t)af;
        *count = 1;

        return VOS_SUCCESS;
    }

    /* Copy node name to null terminated string. */
    if (strlen(nodename) >= VOS_MAX_HOSTNAME)
    {
        return VOS_ENAMETOOLONG;
    }

    /* Call getaddrinfo() */
    vos_bzero(&hint, sizeof(hint));
    hint.ai_family = af;

    rc = getaddrinfo(nodename, NULL, &hint, &res);
    if (rc != 0)
        return VOS_ERESOLVE;

    orig_res = res;

    /* Enumerate each item in the result */
    for (i = 0; i<*count && res; res = res->ai_next)
    {
        /* Ignore unwanted address families */
        if (af != AF_UNSPEC && res->ai_family != af)
            continue;

        /* Store canonical name (possibly truncating the name) */
        if (res->ai_canonname)
        {
            vos_ansi_strncpy(ai[i].ai_canonname, res->ai_canonname,
                             sizeof(ai[i].ai_canonname));
            ai[i].ai_canonname[sizeof(ai[i].ai_canonname) - 1] = '\0';
        }
        else
        {
            vos_ansi_strcpy(ai[i].ai_canonname, nodename);
        }

        /* Store address */
        VOS_ASSERT_ON_FAIL(res->ai_addrlen <= sizeof(vos_sockaddr), continue);
        memcpy(&ai[i].ai_addr, res->ai_addr, res->ai_addrlen);
        VOS_SOCKADDR_RESET_LEN(&ai[i].ai_addr);

        /* Next slot */
        ++i;
    }

    *count = i;

    freeaddrinfo(orig_res);

    /* Done */
    return VOS_SUCCESS;

#else	/* VOS_SOCK_HAS_GETADDRINFO */
    vos_bool_t has_addr = FALSE;

    VOS_ASSERT_RETURN(count && *count, VOS_EINVAL);

    /* Check if nodename is IP address */
    vos_bzero(&ai[0], sizeof(ai[0]));
    if ((af == AF_INET || af == AF_UNSPEC) &&
        vos_inet_pton(AF_INET, nodename,
                      &ai[0].ai_addr.ipv4.sin_addr) == VOS_SUCCESS)
    {
        af = AF_INET;
        has_addr = TRUE;
    }
    else if ((af == AF_INET6 || af == AF_UNSPEC) &&
             vos_inet_pton(AF_INET6, nodename,
                           &ai[0].ai_addr.ipv6.sin6_addr) == VOS_SUCCESS)
    {
        af = AF_INET6;
        has_addr = TRUE;
    }

    if (has_addr)
    {
        vos_str_t tmp, name;

        name = vos_str((char*)nodename);
        tmp.ptr = ai[0].ai_canonname;

        vos_strncpy_with_null(&tmp, &name, VOS_MAX_HOSTNAME);
        ai[0].ai_addr.addr.sa_family = (vos_uint16_t)af;
        *count = 1;

        return VOS_SUCCESS;
    }

    if (af == AF_INET || af == AF_UNSPEC)
    {
        vos_hostent he;
        unsigned i, max_count;
        vos_status_t status;

        //#ifdef _MSC_VER
        vos_bzero(&he, sizeof(he));
        //#endif

        status = vos_gethostbyname(nodename, &he);
        if (status != VOS_SUCCESS)
            return status;

        max_count = *count;
        *count = 0;

        vos_bzero(ai, max_count * sizeof(vos_addrinfo));

        for (i = 0; he.h_addr_list[i] && *count<max_count; ++i)
        {
            vos_ansi_strncpy(ai[*count].ai_canonname, he.h_name, sizeof(ai[*count].ai_canonname));
            ai[*count].ai_canonname[sizeof(ai[*count].ai_canonname) - 1] = '\0';

            ai[*count].ai_addr.ipv4.sin_family = AF_INET;
            memcpy(&ai[*count].ai_addr.ipv4.sin_addr, he.h_addr_list[i], he.h_length);
            VOS_SOCKADDR_RESET_LEN(&ai[*count].ai_addr);

            (*count)++;
        }

        return VOS_SUCCESS;

    }
    else
    {
        /* IPv6 is not supported */
        *count = 0;

        return -1;// VOS_EIPV6NOTSUP;
    }
#endif	/* VOS_SOCK_HAS_GETADDRINFO */
}

vos_status_t vos_gethostip(int af, vos_sockaddr *addr)
{
    unsigned i, count, cand_cnt;
    enum
    {
        CAND_CNT = 8,
        /* Weighting to be applied to found addresses */
        WEIGHT_HOSTNAME = 1,	/* hostname IP is not always valid! */
        WEIGHT_DEF_ROUTE = 2,
        WEIGHT_INTERFACE = 1,
        WEIGHT_LOOPBACK = -5,
        WEIGHT_LINK_LOCAL = -4,
        WEIGHT_DISABLED = -50,
        MIN_WEIGHT = WEIGHT_DISABLED + 1	/* minimum weight to use */
    };

    /* candidates: */
    vos_sockaddr cand_addr[CAND_CNT];
    int		cand_weight[CAND_CNT];
    int	        selected_cand;
    char	strip[VOS_INET6_ADDRSTRLEN + 10];

    /* Special IPv4 addresses. */
    struct spec_ipv4_t
    {
        vos_uint32_t addr;
        vos_uint32_t mask;
        int	    weight;
    } spec_ipv4[] =
    {
        /* 127.0.0.0/8, loopback addr will be used if there is no other
        * addresses.
        */
        { 0x7f000000, 0xFF000000, WEIGHT_LOOPBACK },

        /* 0.0.0.0/8, special IP that doesn't seem to be practically useful */
        { 0x00000000, 0xFF000000, WEIGHT_DISABLED },

        /* 169.254.0.0/16, a zeroconf/link-local address, which has higher
        * priority than loopback and will be used if there is no other
        * valid addresses.
        */
        { 0xa9fe0000, 0xFFFF0000, WEIGHT_LINK_LOCAL }
    };

    /* Special IPv6 addresses */
    struct spec_ipv6_t
    {
        vos_uint8_t addr[16];
        vos_uint8_t mask[16];
        int	   weight;
    } spec_ipv6[] =
    {
        /* Loopback address, ::1/128 */
        { { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 },
        { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
        WEIGHT_LOOPBACK
        },

        /* Link local, fe80::/10 */
        { { 0xfe,0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0xff,0xc0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        WEIGHT_LINK_LOCAL
        },

        /* Disabled, ::/128 */
        { { 0x0,0x0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
        WEIGHT_DISABLED
        }
    };

    vos_addrinfo ai;
    vos_status_t status;
    char hostname[VOS_MAX_HOSTNAME + 1];

    /* May not be used if TRACE_ is disabled */
    VOS_UNUSED_ARG(strip);

#ifdef _MSC_VER
    /* Get rid of "uninitialized he variable" with MS compilers */
    vos_bzero(&ai, sizeof(ai));
#endif

    cand_cnt = 0;
    vos_bzero(cand_addr, sizeof(cand_addr));
    vos_bzero(cand_weight, sizeof(cand_weight));
    for (i = 0; i<VOS_ARRAY_SIZE(cand_addr); ++i)
    {
        cand_addr[i].addr.sa_family = (vos_uint16_t)af;
        VOS_SOCKADDR_RESET_LEN(&cand_addr[i]);
    }

    addr->addr.sa_family = (vos_uint16_t)af;
    VOS_SOCKADDR_RESET_LEN(addr);

#if !defined(VOS_GETHOSTIP_DISABLE_LOCAL_RESOLUTION) || \
    VOS_GETHOSTIP_DISABLE_LOCAL_RESOLUTION == 0
    /* Get hostname's IP address */
    count = 1;

    vos_str2(vos_gethostname(), hostname, VOS_MAX_HOSTNAME);

    status = vos_getaddrinfo(af, hostname, &count, &ai);
    if (status == VOS_SUCCESS)
    {
        vos_assert(ai.ai_addr.addr.sa_family == (vos_uint16_t)af);
        vos_sockaddr_copy_addr(&cand_addr[cand_cnt], &ai.ai_addr);
        vos_sockaddr_set_port(&cand_addr[cand_cnt], 0);
        cand_weight[cand_cnt] += WEIGHT_HOSTNAME;
        ++cand_cnt;

        VOS_LOG(1, (THIS_FILE, "hostname IP is %s", vos_sockaddr_print(&ai.ai_addr, strip, sizeof(strip), 0)));
    }
#else
    PJ_UNUSED_ARG(ai);
#endif

    /* Get default interface (interface for default route) */
    if (cand_cnt < VOS_ARRAY_SIZE(cand_addr))
    {
        status = vos_getdefaultipinterface(af, addr);
        if (status == VOS_SUCCESS)
        {
            VOS_LOG(1, (THIS_FILE, "default IP is %s", vos_sockaddr_print(addr, strip, sizeof(strip), 0)));
            vos_sockaddr_set_port(addr, 0);
            for (i = 0; i<cand_cnt; ++i)
            {
                if (vos_sockaddr_cmp(&cand_addr[i], addr) == 0)
                    break;
            }

            cand_weight[i] += WEIGHT_DEF_ROUTE;
            if (i >= cand_cnt)
            {
                vos_sockaddr_copy_addr(&cand_addr[i], addr);
                ++cand_cnt;
            }
        }
    }


    /* Enumerate IP interfaces */
    if (cand_cnt < VOS_ARRAY_SIZE(cand_addr))
    {
        unsigned start_if = cand_cnt;
        count = VOS_ARRAY_SIZE(cand_addr) - start_if;

        status = vos_enum_ip_interface(af, &count, &cand_addr[start_if]);
        if (status == VOS_SUCCESS && count)
        {
            /* Clear the port number */
            for (i = 0; i<count; ++i)
                vos_sockaddr_set_port(&cand_addr[start_if + i], 0);

            /* For each candidate that we found so far (that is the hostname
            * address and default interface address, check if they're found
            * in the interface list. If found, add the weight, and if not,
            * decrease the weight.
            */
            for (i = 0; i<cand_cnt; ++i)
            {
                unsigned j;
                for (j = 0; j<count; ++j)
                {
                    if (vos_sockaddr_cmp(&cand_addr[i],
                                         &cand_addr[start_if + j]) == 0)
                        break;
                }

                if (j == count)
                {
                    /* Not found */
                    cand_weight[i] -= WEIGHT_INTERFACE;
                }
                else
                {
                    cand_weight[i] += WEIGHT_INTERFACE;
                }
            }

            /* Add remaining interface to candidate list. */
            for (i = 0; i<count; ++i)
            {
                unsigned j;
                for (j = 0; j<cand_cnt; ++j)
                {
                    if (vos_sockaddr_cmp(&cand_addr[start_if + i],
                                         &cand_addr[j]) == 0)
                        break;
                }

                if (j == cand_cnt)
                {
                    vos_sockaddr_copy_addr(&cand_addr[cand_cnt],
                                           &cand_addr[start_if + i]);
                    cand_weight[cand_cnt] += WEIGHT_INTERFACE;
                    ++cand_cnt;
                }
            }
        }
    }

    /* Apply weight adjustment for special IPv4/IPv6 addresses
    */
    if (af == AF_INET)
    {
        for (i = 0; i<cand_cnt; ++i)
        {
            unsigned j;
            for (j = 0; j<VOS_ARRAY_SIZE(spec_ipv4); ++j)
            {
                vos_uint32_t a = vos_ntohl(cand_addr[i].ipv4.sin_addr.s_addr);
                vos_uint32_t pa = spec_ipv4[j].addr;
                vos_uint32_t pm = spec_ipv4[j].mask;

                if ((a & pm) == pa)
                {
                    cand_weight[i] += spec_ipv4[j].weight;
                    break;
                }
            }
        }
    }
    else if (af == AF_INET6)
    {
        for (i = 0; i<VOS_ARRAY_SIZE(spec_ipv6); ++i)
        {
            unsigned j;
            for (j = 0; j<cand_cnt; ++j)
            {
                vos_uint8_t *a = cand_addr[j].ipv6.sin6_addr.s6_addr;
                vos_uint8_t am[16];
                vos_uint8_t *pa = spec_ipv6[i].addr;
                vos_uint8_t *pm = spec_ipv6[i].mask;
                unsigned k;

                for (k = 0; k<16; ++k)
                {
                    am[k] = (vos_uint8_t)((a[k] & pm[k]) & 0xFF);
                }

                if (memcmp(am, pa, 16) == 0)
                {
                    cand_weight[j] += spec_ipv6[i].weight;
                }
            }
        }
    }
    else
    {
        return -1;// VOS_EAFNOTSUP;
    }

    /* Enumerate candidates to get the best IP address to choose */
    selected_cand = -1;
    for (i = 0; i<cand_cnt; ++i)
    {
        VOS_LOG(1, (THIS_FILE, "Checking candidate IP %s, weight=%d", vos_sockaddr_print(&cand_addr[i], strip, sizeof(strip), 0), cand_weight[i]));

        if (cand_weight[i] < MIN_WEIGHT)
        {
            continue;
        }

        if (selected_cand == -1)
        {
            selected_cand = i;
        }
        else if (cand_weight[i] > cand_weight[selected_cand])
        {
            selected_cand = i;
        }
    }

    /* If else fails, returns loopback interface as the last resort */
    if (selected_cand == -1)
    {
        if (af == AF_INET)
        {
            addr->ipv4.sin_addr.s_addr = vos_htonl(0x7f000001);
        }
        else
        {
            vos_in6_addr *s6_addr;

            s6_addr = (vos_in6_addr*)vos_sockaddr_get_addr(addr);
            vos_bzero(s6_addr, sizeof(vos_in6_addr));
            s6_addr->s6_addr[15] = 1;
        }
        VOS_LOG(1, (THIS_FILE, "Loopback IP %s returned", vos_sockaddr_print(addr, strip, sizeof(strip), 0)));
    }
    else
    {
        vos_sockaddr_copy_addr(addr, &cand_addr[selected_cand]);
        VOS_LOG(1, (THIS_FILE, "Candidate %s selected", vos_sockaddr_print(addr, strip, sizeof(strip), 0)));
    }

    return VOS_SUCCESS;
}
