#define __DNL_LOG_C__

#include <stdarg.h>
#include <string.h>
#include "dnl_log.h"
#include "dnl_dev.h"
#include "vos_assert.h"
#include "vos_string.h"

#if (OS_UCOS_II == 1)
void dnl_log_print(const char *format, ...)
{
    va_list arg;
    char fmt[128];
    const char *filename;
    
    memset(fmt, 0x0, sizeof(fmt));
    snprintf(fmt, sizeof(fmt), "%s\n", format);
    va_start(arg, format);
    fhprintf(fmt, arg);
    va_end(arg);
}

#else

#include <stdarg.h>
#include "vos_file_access.h"
#include "vos_file_io.h"

static char             g_logfile_fullname[256]={0};
static vos_oshandle_t	g_logfile_fd = NULL;
static vos_mutex_t	    *g_log_lock = NULL;

#define THIS_FILE	"dnl_log.c"

static vos_status_t dnl_logfile_switch();
static void dnl_log_write(int level, const char *buffer, int len);


static void dnl_log_write(int level, const char *buffer, int len)
{
	vos_ssize_t size = len;
	//vos_file_stat stat;

	VOS_UNUSED_ARG(level);
	VOS_UNUSED_ARG(len);

	if (g_logfile_fd) 
	{
        vos_offset_t file_size;
		vos_mutex_lock(g_log_lock);
		vos_file_write(g_logfile_fd, buffer, &size);
		
		vos_file_flush(g_logfile_fd);

		//vos_file_getstat(g_logfile_fullname, &stat);
        //if( stat.size/1024 > DNL_LOG_SWITCH_SIZE)
        //{
        //    dnl_logfile_switch();
        //}
        vos_file_getpos(g_logfile_fd, &file_size);
        if( file_size/1024 > (g_DnlDevInfo.log_max_size?g_DnlDevInfo.log_max_size:DNL_LOG_SWITCH_SIZE ) )
        {
            dnl_logfile_switch();
        }
        
		vos_mutex_unlock(g_log_lock);
	}
}

static vos_status_t dnl_logfile_switch()
{
	vos_status_t status;
	vos_time_val tv;
	vos_parsed_time pt;
	char newname[128];

	unsigned flags = VOS_O_WRONLY|VOS_O_RDONLY|VOS_O_APPEND;

	if(g_logfile_fd)
	{
		vos_file_close(g_logfile_fd);
		g_logfile_fd = NULL;
	}

	//if(OS_WIN32 == 1 && vos_file_exists(g_logfile_fullname, FALSE))
    if( vos_file_exists(g_logfile_fullname, FALSE) )
	{
		if( g_DnlDevInfo.log_backup_flag )
		{
			vos_gettimeofday(&tv);
			vos_time_decode(&tv, &pt);
			memset(newname, 0x0, sizeof(newname));
			vos_native_snprintf(newname, sizeof(newname), "%s.%04d%02d%02d%02d%02d", 
				g_logfile_fullname, pt.year, pt.mon+1, pt.hour, pt.min, pt.sec);

			vos_file_move(g_logfile_fullname, newname);
		}
		else
		{
			vos_file_delete(g_logfile_fullname);
		}
	}

	if(vos_ansi_strlen(g_logfile_fullname))
	{
		status = vos_file_open(g_logfile_fullname, flags, &g_logfile_fd);
		if (status != VOS_SUCCESS) 
		{
			VOS_LOG(1, (THIS_FILE, "Error creating log file", status));
			return status;
		}
	}
	return VOS_SUCCESS;
}

vos_status_t dnl_log_init()
{
    char logfile_name[128]={"dnl.log"};
    if( strlen(g_DnlDevInfo.dev_id) > 0 )
    {
        int len = snprintf(logfile_name, sizeof(logfile_name)-1, "dnl_%s.log", g_DnlDevInfo.dev_id);
        if( len > 0 && len < sizeof(g_logfile_fullname)-1 )
        {
            logfile_name[len] = '\0';
        }
    }

    memset(g_logfile_fullname, 0, sizeof(g_logfile_fullname));

    if( vos_file_exists(g_DnlDevInfo.log_path, TRUE) )
    {
        int len = strlen(g_DnlDevInfo.log_path);
        if( g_DnlDevInfo.log_path[len-1] == '\\' )
        {
            len = snprintf(g_logfile_fullname, sizeof(g_logfile_fullname)-1, "%s%s", g_DnlDevInfo.log_path, logfile_name);
            if( len > 0 && len < sizeof(g_logfile_fullname)-1 )
            {
                g_logfile_fullname[len] = '\0';
            }
        }
        else
        {
            len = snprintf(g_logfile_fullname, sizeof(g_logfile_fullname)-1, "%s/%s", g_DnlDevInfo.log_path, logfile_name);
            if( len > 0 && len < sizeof(g_logfile_fullname)-1 )
            {
                g_logfile_fullname[len] = '\0';
            }
        }
    }

    if( strlen(g_logfile_fullname) == 0 )
    {
        strcpy(g_logfile_fullname, logfile_name);
    }

    if( g_DnlDevInfo.log_level <= VOS_LOG_LEVEL_TRACE )
    {
        vos_log_set_level(g_DnlDevInfo.log_level);
    }
    else
    {
        vos_log_set_level(VOS_LOG_LEVEL_DEBUG); //VOS_LOG_LEVEL_TRACE
    }
    
	dnl_log_flush_timer = 3;		/*3秒 */
	//dnl_log_debug_enable = FALSE; /* 初始化不开启debug日志 */
    vos_mutex_create("dnl_log_lock", 0, &g_log_lock);
	vos_log_set_log_func(&dnl_log_write);

	return dnl_logfile_switch();
}

void dnl_log_flush()
{	
	if (g_logfile_fd) 
	{
		vos_file_flush(g_logfile_fd);
	}

	dnl_logfile_switch();
}

void dnl_log_print(int level, const char* file, int line, const char *format, ...)
{	
	va_list arg;
	char sender[128];
	const char *filename;
	
	filename = file + strlen(file);
	while( filename != file)
	{
		if( (*filename == '\\') || (*filename == '/') )
		{
			filename ++;
			break;
		}
		filename --;
	}

	memset(sender, 0x0, sizeof(sender));
	vos_ansi_snprintf(sender, sizeof(sender), "%s:%d", filename, line);

	va_start(arg, format);
	vos_log(sender, level, format, arg);
	va_end(arg);
}

void dnl_log_1s_timeout()
{
	if(dnl_log_flush_timer > 0)
	{
		dnl_log_flush_timer --;
	}
	else
	{
		dnl_log_flush_timer = 3;
	}
}

vos_bool_t dnl_log_flush_enable()
{
	if(dnl_log_flush_timer>0)
	{
		return FALSE;
	}

	return TRUE;
}

#endif