
#ifndef __VOS_LOG_H__
#define __VOS_LOG_H__

#include "vos_types.h"
#include <stdarg.h>

#undef  EXT
#ifdef __LOG_C__
#define EXT 
#else
#define EXT extern
#endif

VOS_BEGIN_DECL

enum vos_log_decoration
{
    VOS_LOG_HAS_DAY_NAME        =    1, /**< Include day name [default: no] 	      */
    VOS_LOG_HAS_YEAR            =    2, /**< Include year digit [no]		      */
    VOS_LOG_HAS_MONTH           =    4, /**< Include month [no]		      */
    VOS_LOG_HAS_DAY_OF_MON      =    8, /**< Include day of month [no]	      */
    VOS_LOG_HAS_TIME            =   16, /**< Include time [yes]		      */
    VOS_LOG_HAS_MICRO_SEC       =   32, /**< Include microseconds [yes]             */
    VOS_LOG_HAS_SENDER          =   64, /**< Include sender in the log [yes] 	      */
    VOS_LOG_HAS_NEWLINE         =  128, /**< Terminate each call with newline [yes] */
    VOS_LOG_HAS_CR              =  256, /**< Include carriage return [no] 	      */
    VOS_LOG_HAS_SPACE           =  512, /**< Include two spaces before log [yes]    */
    VOS_LOG_HAS_COLOR           = 1024, /**< Colorize logs [yes on win32]	      */
    VOS_LOG_HAS_LEVEL_TEXT      = 2048, /**< Include level text string [no]	      */
    VOS_LOG_HAS_THREAD_ID       = 4096, /**< Include thread identification [no]     */
    VOS_LOG_HAS_THREAD_SWC      = 8192, /**< Add mark when thread has switched [yes]*/
    VOS_LOG_HAS_INDENT          =16384  /**< Indentation. Say yes! [yes]            */
};

enum vos_log_level
{
    VOS_LOG_LEVEL_FATAL = 0,
    VOS_LOG_LEVEL_ERR = 1,
    VOS_LOG_LEVEL_WAR = 2,
    VOS_LOG_LEVEL_INFO = 3,
	VOS_LOG_LEVEL_DEBUG = 4,
    VOS_LOG_LEVEL_TRACE = 5,
    VOS_LOG_LEVEL_DETRC = 6, 
    VOS_LOG_LEVEL_INVALID         
};

typedef void vos_log_func(int level, const char *data, int len);

#define VOS_LOG(level,arg)	do { \
												if (level <= vos_log_get_level()) \
												vos_log_wrapper_##level(arg); \
											} while (0)
/*
#if VOS_LOG_MAX_LEVEL >= 0
#  define vos_log_wrapper_0(arg)	vos_log_0 arg
#else
#  define vos_log_wrapper_0(arg)
#endif
*/
#if VOS_LOG_MAX_LEVEL >= 1
#  define vos_log_wrapper_1(arg)	vos_log_1 arg
#else
#  define vos_log_wrapper_1(arg)
#endif
#if VOS_LOG_MAX_LEVEL >= 2
#  define vos_log_wrapper_2(arg)	vos_log_2 arg
#else
#  define vos_log_wrapper_2(arg)
#endif
#if VOS_LOG_MAX_LEVEL >= 3
#  define vos_log_wrapper_3(arg)	vos_log_3 arg
#else
#  define vos_log_wrapper_3(arg)
#endif
#if VOS_LOG_MAX_LEVEL >= 4
#  define vos_log_wrapper_4(arg)	vos_log_4 arg
#else
#  define vos_log_wrapper_4(arg)
#endif
#if VOS_LOG_MAX_LEVEL >= 5
#  define vos_log_wrapper_5(arg)	vos_log_5 arg
#else
#  define vos_log_wrapper_5(arg)
#endif
#if VOS_LOG_MAX_LEVEL >= 6
#  define vos_log_wrapper_6(arg)	vos_log_6 arg
#else
#  define vos_log_wrapper_6(arg)
#endif

EXT vos_status_t vos_log_init(void);
EXT void vos_log_shutdown();
EXT void vos_log(const char *sender, int level, const char *format, va_list marker);

EXT void vos_log_set_level(int level);
EXT int vos_log_get_level(void);
EXT void vos_log_set_decor(unsigned decor);
//EXT unsigned vos_log_get_decor(void);

EXT void vos_log_add_indent(int indent);
EXT void vos_log_push_indent(void);
EXT void vos_log_pop_indent(void);
EXT void vos_log_set_color(int level, vos_color_t color);
EXT vos_color_t vos_log_get_color(int level);

EXT void vos_log_set_log_func( vos_log_func *func );
EXT vos_log_func* vos_log_get_log_func(void);

EXT void vos_log_0(const char *src, const char *format, ...);
EXT void vos_log_1(const char *src, const char *format, ...);
EXT void vos_log_2(const char *src, const char *format, ...);
EXT void vos_log_3(const char *src, const char *format, ...);
EXT void vos_log_4(const char *src, const char *format, ...);
EXT void vos_log_5(const char *src, const char *format, ...);
EXT void vos_log_6(const char *src, const char *format, ...);

VOS_END_DECL 

#endif  /* __VOS_LOG_H__ */


