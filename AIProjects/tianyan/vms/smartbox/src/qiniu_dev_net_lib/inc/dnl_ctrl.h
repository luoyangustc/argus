#ifndef __DNL_CTRL_H__
#define __DNL_CTRL_H__

#include "comm_includes.h"

#undef  EXT
#ifndef __DNL_CTRL_C__
#define EXT extern
#else
#define EXT
#endif

#define ENTRY_TASK_STACK_SIZE        (512*2 - 128)
#define STREAM_TASK_STACK_SIZE       512*2
#define OTHER_TASK_STACK_SIZE        512

#define DNL_STREAM_TASK_PRIO        APP_MYUART_PRIO + 4
#define DNL_PROBE_TASK_PRIO         APP_MYUART_PRIO + 2
#define DNL_CMD_TASK_PRIO           APP_MYUART_PRIO + 3
#define DNL_ENTRY_TASK_PRIO         APP_MYUART_PRIO + 1		//43+1
#define DNL_STREAM_REPORT_PRIO      32//26 //32

typedef enum dnl_thread_index_e
{
    DNL_THD_IDX_ENTRY,
    DNL_THD_IDX_STREAM,
	DNL_THD_IDX_APP_CB,
    DNL_THD_IDX_PROBE,
	DNL_THD_IDX_EVENT,
    DNL_THD_IDX_TIMER,
    DNL_THD_IDX_MAX
}dnl_thread_index_e;

typedef enum dnl_result_e
{
	RS_OK,
	RS_NG,
	RS_KP,
	RS_END
}dnl_result_e;

typedef struct dnl_conttbl_t
{
	int				(*func)(void*);
	int				ok_seq;
	int				ng_seq;
	int				cont;
}dnl_conttbl_t;

//EXT vos_timer_heap_t*		g_DnlTimerHeap;
//EXT vos_timer_entry*		g_DnlTimerEntry;
EXT vos_thread_t*			g_DnlThreadID[DNL_THD_IDX_MAX];

EXT vos_status_t dnl_init(void);
EXT void dnl_destroy(void);

#endif
