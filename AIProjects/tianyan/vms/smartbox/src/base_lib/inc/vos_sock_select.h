#ifndef __VOS_SELECT_H__
#define __VOS_SELECT_H__

#include "vos_types.h"

#undef  EXT
#ifndef __SELECT_C__
#define EXT    extern
#else
#define EXT
#endif

VOS_BEGIN_DECL 

typedef struct vos_fd_set_t
{
    vos_sock_t data[VOS_SOCK_MAX_HANDLES+ 4]; /**< Opaque buffer for fd_set */
} vos_fd_set_t;

EXT void VOS_FD_ZERO(vos_fd_set_t *fdsetp);
EXT vos_size_t VOS_FD_COUNT(const vos_fd_set_t *fdsetp);
EXT void VOS_FD_SET(vos_sock_t fd, vos_fd_set_t *fdsetp);
EXT void VOS_FD_CLR(vos_sock_t fd, vos_fd_set_t *fdsetp);
EXT vos_bool_t VOS_FD_ISSET(vos_sock_t fd, const vos_fd_set_t *fdsetp);
EXT int vos_sock_select( int n, 
			     vos_fd_set_t *readfds, 
			     vos_fd_set_t *writefds,
			     vos_fd_set_t *exceptfds, 
			     const vos_time_val *timeout);


VOS_END_DECL

#endif	/* __VOS_SELECT_H__ */

