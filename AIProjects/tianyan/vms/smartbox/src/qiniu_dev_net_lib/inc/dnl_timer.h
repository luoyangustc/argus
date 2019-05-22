#ifndef __DNL_TIMER_H__
#define __DNL_TIMER_H__

#include "comm_includes.h"

#undef  EXT
#ifndef __DNL_TIMER_C__
#define EXT extern
#else
#define EXT
#endif

#define  TIME_OUT 0

EXT int mytimer_init(void);
EXT int dnl_start_timer();
EXT void dnl_stop_timer();

#endif
