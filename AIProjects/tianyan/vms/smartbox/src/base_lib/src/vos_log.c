#define __LOG_C__

#include "vos_log.h"
#include "vos_time.h"
#include "vos_os.h"
#include "vos_string.h"

///////////////////////////////////////////////////////////////

//log writer stdout

///////////////////////////////////////////////////////////////

static void term_set_color(int level)
{
#if defined(VOS_TERM_HAS_COLOR) && VOS_TERM_HAS_COLOR != 0
	vos_term_set_color(vos_log_get_color(level));
#else
	VOS_UNUSED_ARG(level);
#endif
}

static void term_restore_color(void)
{
#if defined(VOS_TERM_HAS_COLOR) && VOS_TERM_HAS_COLOR != 0
	vos_term_set_color(vos_log_get_color(77));
#endif
}

static void vos_log_write_std(int level, const char *buffer, int len)
{
	//    //VOS_CHECK_STACK;
	VOS_UNUSED_ARG(len);

	/* Copy to terminal/file. */
	//if (vos_log_get_decor() & VOS_LOG_HAS_COLOR)
    if(0)
	{
		term_set_color(level);
		printf("%s", buffer);
		term_restore_color();
	}
	else
	{
		printf("%s", buffer);
	}
}

/////////////////////////////////////////////////////////////////

//string

/////////////////////////////////////////////////////////////////

int vos_utoa_pad(unsigned long val, char *buf, int min_dig, int pad)
{
	char *p;
	int len;

	p = buf;
	do
	{
		unsigned long digval = (unsigned long)(val % 10);
		val /= 10;
		*p++ = (char)(digval + '0');
	} while (val > 0);

	len = p - buf;
	while (len < min_dig)
	{
		*p++ = (char)pad;
		++len;
	}
	*p-- = '\0';

	do
	{
		char temp = *p;
		*p = *buf;
		*buf = temp;
		--p;
		++buf;
	} while (buf < p);

	return len;
}

int vos_utoa(unsigned long val, char *buf)
{
	return vos_utoa_pad(val, buf, 0, 0);
}

//默认日志打印函数
//extern void vos_log_write_std(int level, const char *buffer, int len);

static int vos_log_max_level = VOS_LOG_MAX_LEVEL;
static vos_thread_id_t g_last_thread_id;
static vos_mutex_t* g_log_mutex;

static vos_log_func *log_writer = &vos_log_write_std;
static unsigned log_decor = 
                VOS_LOG_HAS_LEVEL_TEXT| VOS_LOG_HAS_YEAR | VOS_LOG_HAS_MONTH | VOS_LOG_HAS_DAY_OF_MON | VOS_LOG_HAS_TIME | 
                VOS_LOG_HAS_MICRO_SEC | VOS_LOG_HAS_SENDER | VOS_LOG_HAS_NEWLINE | VOS_LOG_HAS_THREAD_ID |
			    VOS_LOG_HAS_SPACE | VOS_LOG_HAS_THREAD_SWC | VOS_LOG_HAS_INDENT;
#if VOS_LOG_USE_STACK_BUFFER==0
static char log_buffer[VOS_LOG_MAX_SIZE];
#endif

#define LOG_MAX_INDENT		80

vos_status_t vos_log_init(void)
{
    g_last_thread_id = 0;
    if( vos_mutex_create("log_mutex", 0, &g_log_mutex) != VOS_SUCCESS )
    {
        return VOS_EINVAL;
    }
    return VOS_SUCCESS;
}

void vos_log_shutdown(void)
{
    vos_mutex_destroy(g_log_mutex);
}

void vos_log_set_level(int level)
{
    vos_log_max_level = level;
}

int vos_log_get_level(void)
{
    return vos_log_max_level;
}

void vos_log_set_log_func( vos_log_func *func )
{
    log_writer = func;
}

vos_log_func* vos_log_get_log_func(void)
{
    return log_writer;
}

void vos_log( const char *sender, int level, 
		     const char *format, va_list marker)
{
    vos_time_val now;
    vos_parsed_time ptime;
    char *pre;
#if VOS_LOG_USE_STACK_BUFFER
    char log_buffer[VOS_LOG_MAX_SIZE];
#endif
	int /*saved_level,*/ len, print_len;// , indent;

    if (level > vos_log_max_level)
    {
	    return;
    }
    
    vos_gettimeofday(&now);
    vos_time_decode(&now, &ptime);

    pre = log_buffer;    
    if (log_decor & VOS_LOG_HAS_DAY_NAME) 
    {
	    static const char *wdays[] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
	    vos_ansi_strcpy(pre, wdays[ptime.wday]);
	    pre += 3;
    }
    
    if (log_decor & VOS_LOG_HAS_YEAR) 
    {
	    if (pre!=log_buffer) *pre++ = ' ';
	    pre += vos_utoa(ptime.year, pre);
    }

    if (log_decor & VOS_LOG_HAS_MONTH) 
    {
	    *pre++ = '-';
	    pre += vos_utoa_pad(ptime.mon+1, pre, 2, '0');
    }
    
    if (log_decor & VOS_LOG_HAS_DAY_OF_MON) 
    {
	    *pre++ = '-';
	    pre += vos_utoa_pad(ptime.day, pre, 2, '0');
    }

    if (log_decor & VOS_LOG_HAS_TIME) 
    {
	    if (pre!=log_buffer) *pre++ = ' ';
	    pre += vos_utoa_pad(ptime.hour, pre, 2, '0');
	    *pre++ = ':';
	    pre += vos_utoa_pad(ptime.min, pre, 2, '0');
	    *pre++ = ':';
	    pre += vos_utoa_pad(ptime.sec, pre, 2, '0');
    }

    if (log_decor & VOS_LOG_HAS_MICRO_SEC) 
    {
	    *pre++ = '.';
	    pre += vos_utoa_pad(ptime.msec, pre, 3, '0');
    }

    if (log_decor & VOS_LOG_HAS_LEVEL_TEXT) 
    {
	    static const char *ltexts[] = { "FATAL:", "  ERR:", "  WAR:", " INFO:", "  DBG:", "TRACE:", "DETRC:"};
	    vos_ansi_strcpy(pre, ltexts[level]);
	    pre += 6;
    }
    
    if (log_decor & VOS_LOG_HAS_SENDER)
    {
        if (pre!=log_buffer) 
        {
            *pre++ = ' ';
        }
        
		while (*sender)
			*pre++ = *sender++;
    }

    if (log_decor & VOS_LOG_HAS_THREAD_ID) 
    {
        vos_thread_id_t thread_id = vos_get_cur_thread_id();
        *pre++ = ' ';
        pre += vos_utoa_pad(thread_id, pre, 10, '0');
    }

    if (log_decor != 0 && log_decor != VOS_LOG_HAS_NEWLINE)
    {
	    *pre++ = ' ';
    }

    if (log_decor & VOS_LOG_HAS_THREAD_SWC) 
    {
	    vos_thread_id_t cur_thread_id = vos_get_cur_thread_id();
	    if (cur_thread_id != g_last_thread_id) 
        {
	        *pre++ = '!';
	        g_last_thread_id = cur_thread_id;
	    } 
        else 
        {
	        *pre++ = ' ';
	    }
    } 
    else if (log_decor & VOS_LOG_HAS_SPACE) 
    {
	    *pre++ = ' ';
    }

    len = pre - log_buffer;

    print_len = vos_ansi_vsnprintf(pre, sizeof(log_buffer)-len, format, marker);
    if (print_len < 0) 
    {
	    level = 1;
	    print_len = vos_ansi_snprintf(pre, sizeof(log_buffer)-len, 
				         "<logging error: msg too long>");
    }

    len = len + print_len;
    if (len > 0 && len < (int)sizeof(log_buffer)-2) 
    {
	    if (log_decor & VOS_LOG_HAS_CR) 
        {
	        log_buffer[len++] = '\r';
	    }
	    
	    if (log_decor & VOS_LOG_HAS_NEWLINE) 
        {
	        log_buffer[len++] = '\n';
	    }
	    
	    log_buffer[len] = '\0';
    } 
    else 
    {
	    len = sizeof(log_buffer)-1;
	    if (log_decor & VOS_LOG_HAS_CR) 
        {
	        log_buffer[sizeof(log_buffer)-3] = '\r';
	    }
	    
	    if (log_decor & VOS_LOG_HAS_NEWLINE) 
        {
	        log_buffer[sizeof(log_buffer)-2] = '\n';
	    }
	    
	    log_buffer[sizeof(log_buffer)-1] = '\0';
    }

    if (log_writer)
    {
	    (*log_writer)(level, log_buffer, len);
    }
}

void vos_log_0(const char *obj, const char *format, ...)
{
	va_list arg;
	va_start(arg, format);
	vos_log(obj, 0, format, arg);
	va_end(arg);
}

void vos_log_1(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 1, format, arg);
    va_end(arg);
}

void vos_log_2(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 2, format, arg);
    va_end(arg);
}

void vos_log_3(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 3, format, arg);
    va_end(arg);
}

void vos_log_4(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 4, format, arg);
    va_end(arg);
}

void vos_log_5(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 5, format, arg);
    va_end(arg);
}


void vos_log_6(const char *obj, const char *format, ...)
{
    va_list arg;
    va_start(arg, format);
    vos_log(obj, 6, format, arg);
    va_end(arg);
}


void vos_log_level(int level, const char *obj, const char *format, va_list marker)
{
//	va_list arg;
//	va_start(arg, format);
	vos_log(obj, level, format, marker);
//	va_end(arg);	
}



