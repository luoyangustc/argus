#define __SELECT_C__
#include "vos_assert.h"
#include "vos_string.h"
#include "vos_os.h"
#include "vos_socket.h"
#include "vos_sock_select.h"

#if defined(VOS_HAS_SYS_TIME_H) && VOS_HAS_SYS_TIME_H!=0
#   include <sys/time.h>
#endif

#ifdef _MSC_VER
#   pragma warning(disable: 4018)    // Signed/unsigned mismatch in FD_*
#   pragma warning(disable: 4389)    // Signed/unsigned mismatch in FD_*
#endif

#define PART_FDSET(ps)		    ((fd_set*)&ps->data[1])
#define PART_FDSET_OR_NULL(ps)	(ps ? PART_FDSET(ps) : NULL)
#define PART_COUNT(ps)		    (ps->data[0])

void VOS_FD_ZERO(vos_fd_set_t *fdsetp)
{
//    //VOS_CHECK_STACK;
    vos_assert(sizeof(vos_fd_set_t)-sizeof(vos_sock_t) >= sizeof(fd_set));

    FD_ZERO(PART_FDSET(fdsetp));
    PART_COUNT(fdsetp) = 0;
}


void VOS_FD_SET(vos_sock_t fd, vos_fd_set_t *fdsetp)
{
//    //VOS_CHECK_STACK;
    vos_assert(sizeof(vos_fd_set_t)-sizeof(vos_sock_t) >= sizeof(fd_set));

    if (!VOS_FD_ISSET(fd, fdsetp))
    {
        ++PART_COUNT(fdsetp);
    }
    FD_SET(fd, PART_FDSET(fdsetp));
}


void VOS_FD_CLR(vos_sock_t fd, vos_fd_set_t *fdsetp)
{
//    //VOS_CHECK_STACK;
    vos_assert(sizeof(vos_fd_set_t)-sizeof(vos_sock_t) >= sizeof(fd_set));

    if (VOS_FD_ISSET(fd, fdsetp))
        --PART_COUNT(fdsetp);
    FD_CLR(fd, PART_FDSET(fdsetp));
}


vos_bool_t VOS_FD_ISSET(vos_sock_t fd, const vos_fd_set_t *fdsetp)
{
//    //VOS_CHECK_STACK;
    VOS_ASSERT_RETURN(sizeof(vos_fd_set_t)-sizeof(vos_sock_t) >= sizeof(fd_set),0);

    return FD_ISSET(fd, PART_FDSET(fdsetp));
}

vos_size_t VOS_FD_COUNT(const vos_fd_set_t *fdsetp)
{
    return PART_COUNT(fdsetp);
}

int vos_sock_select( int n, 
			    vos_fd_set_t *readfds, 
			    vos_fd_set_t *writefds,
			    vos_fd_set_t *exceptfds, 
			    const vos_time_val *timeout)
{
    struct timeval os_timeout, *p_os_timeout;

 //   //VOS_CHECK_STACK;

    VOS_ASSERT_RETURN(sizeof(vos_fd_set_t)-sizeof(vos_sock_t) >= sizeof(fd_set),VOS_EINVAL);

    if (timeout) 
    {
    	os_timeout.tv_sec = timeout->sec;
    	os_timeout.tv_usec = timeout->usec * 1000;
    	p_os_timeout = &os_timeout;
    } 
    else 
    {
	    p_os_timeout = NULL;
    }

    return select(n, PART_FDSET_OR_NULL(readfds), PART_FDSET_OR_NULL(writefds),
		  PART_FDSET_OR_NULL(exceptfds), p_os_timeout);
}


