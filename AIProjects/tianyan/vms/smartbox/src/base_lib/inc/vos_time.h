#ifndef __VOS_TIME_H__
#define __VOS_TIME_H__

#include "vos_types.h"
#if defined(VOS_HAS_TIME_H) && VOS_HAS_TIME_H != 0
#  include <time.h>
#endif

#if defined(VOS_HAS_SYS_TIME_H) && VOS_HAS_SYS_TIME_H != 0
#  include <sys/time.h>
#endif

#if defined(VOS_HAS_SYS_TIMEB_H) && VOS_HAS_SYS_TIMEB_H != 0
#  include <sys/timeb.h>
#endif

#undef	EXT
#ifndef __VOS_TIME_C__
#define EXT extern
#else
#define EXT
#endif


EXT vos_status_t vos_gettimeofday(vos_time_val *p_tv);
EXT vos_status_t vos_settimeofday(const vos_time_val *p_tv);

EXT vos_status_t vos_time_encode(const vos_parsed_time *pt, vos_time_val *tv);
EXT vos_status_t vos_time_decode(const vos_time_val *tv, vos_parsed_time *pt);


EXT vos_uint64_t vos_get_system_usec();
EXT vos_uint64_t vos_get_system_msec();
EXT vos_uint64_t vos_get_system_sec();
EXT vos_int32_t vos_set_system_usec(vos_uint64_t us);

EXT vos_uint32_t vos_get_system_tick(); //ms
EXT vos_uint32_t vos_get_system_tick_sec();

#endif	/* __VOS_TIME_H__ */
