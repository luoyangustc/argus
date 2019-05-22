#ifndef __DNL_LOG_H__
#define  __DNL_LOG_H__


#include "comm_includes.h"
#include "vos_log.h"
#if (OS_UCOS_II == 1)
#include "printf.h"
#endif

#undef  EXT
#ifndef __DNL_LOG_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

#define DNL_LOG_BACKUP_FLAG		0	// 0:不备份， 1：备份
#define DNL_LOG_SWITCH_SIZE		250	// 日志切换的文件大小，单位：KB


#if (OS_LINUX==1)//__GNUC__
#define DNL_FATAL_LOG(fmt, args...)		dnl_log_print(VOS_LOG_LEVEL_FATAL, __FILE__, __LINE__, fmt, ##args)
#define DNL_ERROR_LOG(fmt, args...)		dnl_log_print(VOS_LOG_LEVEL_ERR, __FILE__, __LINE__, fmt, ##args)
#define DNL_WARN_LOG(fmt, args...)		dnl_log_print(VOS_LOG_LEVEL_WAR, __FILE__, __LINE__, fmt, ##args)
#define DNL_INFO_LOG(fmt, args...)			dnl_log_print(VOS_LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##args)
#define DNL_DEBUG_LOG(fmt, args...)		dnl_log_print(VOS_LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args)
#define DNL_TRACE_LOG(fmt, args...)		dnl_log_print(VOS_LOG_LEVEL_TRACE, __FILE__, __LINE__, fmt, ##args)

#elif (OS_UCOS_II == 1)	//fhprintf
#define DNL_FATAL_LOG		fhprintf//dnl_log_print
#define DNL_ERROR_LOG		uprintf//dnl_log_print
#define DNL_WARN_LOG		fhprintf//dnl_log_print
#define DNL_INFO_LOG		fhprintf//dnl_log_print
#define DNL_DEBUG_LOG		//uprintf//dnl_log_print
#define DNL_TRACE_LOG		//uprintf//dnl_log_print


#else
#define DNL_FATAL_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_FATAL, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DNL_ERROR_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_ERR, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DNL_WARN_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_WAR, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DNL_INFO_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DNL_DEBUG_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DNL_TRACE_LOG(fmt, ...)		dnl_log_print(VOS_LOG_LEVEL_TRACE, __FILE__, __LINE__, fmt, __VA_ARGS__)
#endif

#if (OS_UCOS_II == 1)
EXT void dnl_log_print(const char *format, ...);
#else
EXT int dnl_log_flush_timer;
EXT vos_bool_t dnl_log_debug_enable;
EXT char dnl_log_path[VOS_MAXPATH + 1];

EXT vos_status_t dnl_log_init();
EXT void dnl_log_flush();
EXT void dnl_log_1s_timeout();
EXT vos_bool_t dnl_log_flush_enable();
EXT void dnl_log_print(int level, const char *func, int line, const char *format, ...);
#endif


VOS_END_DECL

#endif
