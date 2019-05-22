#ifndef __DNL_EVENT_H__
#define __DNL_EVENT_H__

#include "comm_includes.h"

#undef  EXT
#ifndef __DNL_EVENT_C__
#define EXT extern
#else
#define EXT
#endif

EXT int event_proc_run(void* arg);

#endif