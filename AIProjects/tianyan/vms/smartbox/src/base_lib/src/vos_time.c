#define __OS_TIME_C__

#include "vos_string.h"
#include "vos_assert.h"
#include "vos_os.h"
#include "vos_time.h"

#define SECS_TO_100_NS (10000000)
#define SECS_TO_1_US (1000000)

#if defined(VOS_HAS_UNISTD_H) && VOS_HAS_UNISTD_H!=0
#    include <unistd.h>
#endif

#if  (OS_WIN32 == 1)
#include <windows.h>

static LARGE_INTEGER base_time;

static vos_status_t get_base_time(void)
{
	SYSTEMTIME st;
	FILETIME ft;
	vos_status_t status = VOS_SUCCESS;

	memset(&st, 0, sizeof(st));
	st.wYear = 1970;
	st.wMonth = 1;
	st.wDay = 1;
	SystemTimeToFileTime(&st, &ft);

	base_time.LowPart = ft.dwLowDateTime;
	base_time.HighPart = ft.dwHighDateTime;

	return status;
}
#endif

vos_status_t vos_time_decode(const vos_time_val *tv, vos_parsed_time *pt)
{
#if  (OS_WIN32 == 1)
	LARGE_INTEGER li;
	FILETIME ft;
	SYSTEMTIME st;

    FILETIME ft_local;

	li.QuadPart = (LONGLONG)tv->sec*SECS_TO_100_NS;
	li.QuadPart += base_time.QuadPart;

	ft.dwLowDateTime = li.LowPart;
	ft.dwHighDateTime = li.HighPart;
    FileTimeToLocalFileTime(&ft, &ft_local);

	FileTimeToSystemTime(&ft_local, &st);

	pt->year = st.wYear;
	pt->mon = st.wMonth;
	pt->day = st.wDay;
	pt->wday = st.wDayOfWeek;
	pt->hour = st.wHour;
	pt->min = st.wMinute;
	pt->sec = st.wSecond;
	pt->msec = tv->usec/1000;
#else
	struct tm *local_time;
	local_time = localtime((time_t*)&tv->sec);
	pt->year = local_time->tm_year + 1900;
	pt->mon = local_time->tm_mon;
	pt->day = local_time->tm_mday;
	pt->hour = local_time->tm_hour;
	pt->min = local_time->tm_min;
	pt->sec = local_time->tm_sec;
	pt->wday = local_time->tm_wday;
	pt->msec = tv->usec/1000;
#endif
	return VOS_SUCCESS;
}

vos_status_t vos_time_encode(const vos_parsed_time *pt, vos_time_val *tv)
{
#if  (OS_WIN32 == 1)
	SYSTEMTIME st;
	FILETIME ft;
	LARGE_INTEGER li;

	vos_bzero(&st, sizeof(st));
	st.wYear = (vos_uint16_t)pt->year;
	st.wMonth = (vos_uint16_t)(pt->mon + 1);
	st.wDay = (vos_uint16_t)pt->day;
	st.wHour = (vos_uint16_t)pt->hour;
	st.wMinute = (vos_uint16_t)pt->min;
	st.wSecond = (vos_uint16_t)pt->sec;
	st.wMilliseconds = (vos_uint16_t)pt->msec;

	SystemTimeToFileTime(&st, &ft);

	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	li.QuadPart -= base_time.QuadPart;
    li.QuadPart /= SECS_TO_100_NS;
	tv->sec = li.LowPart;
	tv->usec = st.wMilliseconds*1000;
#else
	struct tm local_time;
	local_time.tm_year = pt->year - 1900;
	local_time.tm_mon = pt->mon;
	local_time.tm_mday = pt->day;
	local_time.tm_hour = pt->hour;
	local_time.tm_min = pt->min;
	local_time.tm_sec = pt->sec;
	local_time.tm_isdst = 0;
	tv->sec = mktime(&local_time);
	tv->usec = pt->msec*1000;
#endif

	return VOS_SUCCESS;
}

vos_status_t vos_gettimeofday(vos_time_val *p_tv)
{
#if  (OS_WIN32 == 1)
	SYSTEMTIME st;
	FILETIME ft;
	LARGE_INTEGER li;
	vos_status_t status;

	if (base_time.QuadPart == 0)
	{
		status = get_base_time();
		if (status != VOS_SUCCESS)
		{
			return status;
		}
	}

	/* Standard Win32 GetLocalTime */
	//GetLocalTime(&st);
	GetSystemTimeAsFileTime (&ft);

	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	li.QuadPart -= base_time.QuadPart;

	p_tv->sec = (long)(li.QuadPart/SECS_TO_100_NS);
	p_tv->usec = (li.QuadPart%SECS_TO_100_NS)/10;
#else
	struct timeval the_time;
	int rc;
	rc = gettimeofday(&the_time, NULL);
	if (rc != 0)
		return VOS_RETURN_OS_ERROR(vos_get_native_os_error());

	p_tv->sec = the_time.tv_sec;
	p_tv->usec = the_time.tv_usec;
#endif
	return VOS_SUCCESS;
}

vos_status_t vos_settimeofday(const vos_time_val *p_tv)
{
#if  (OS_WIN32 == 1)
	SYSTEMTIME st;
	FILETIME ft;
	LARGE_INTEGER li;
	//	vos_status_t status;

	vos_assert(p_tv);
	/*	if(!p_tv)
	{
	return -1;
	}
	*/
	li.QuadPart = p_tv->sec*SECS_TO_100_NS;
	li.QuadPart += base_time.QuadPart;

	ft.dwLowDateTime = li.LowPart;
	ft.dwHighDateTime = li.HighPart;
	FileTimeToSystemTime(&ft, &st);
	SetLocalTime(&st);

#else
	struct timeval tv;
	int rc;

	tv.tv_sec = p_tv->sec;
	tv.tv_usec = p_tv->usec * 1000;

	rc = settimeofday(&tv, NULL);
	if (rc != 0)
	{
		return VOS_RETURN_OS_ERROR(vos_get_native_os_error());
	}
#endif
	return VOS_SUCCESS;
}

/////////////////////////////////////////////
//time
/////////////////////////////////////////////
//#include <Mmsystem.h>    
//#pragma comment(lib, "Winmm.lib")  

vos_uint64_t vos_get_system_usec()
{
    vos_time_val tv;
    vos_uint64_t tim_us;
    vos_gettimeofday(&tv);

    tim_us = (vos_uint64_t)tv.sec * SECS_TO_1_US + (vos_uint64_t)tv.usec;

    return tim_us;
}

vos_uint64_t vos_get_system_msec()
{
    vos_time_val tv;
    vos_uint64_t tim_ms;
    vos_gettimeofday(&tv);
    tim_ms = (vos_uint64_t)tv.sec * 1000 + (vos_uint64_t)tv.usec/1000;
    return tim_ms;
}

vos_uint64_t vos_get_system_sec()
{
    vos_time_val tv;
    //vos_uint64_t tim_ms;
    vos_gettimeofday(&tv);
    return (vos_uint64_t)tv.sec;
}

vos_int32_t vos_set_system_usec(vos_uint64_t us)
{
    vos_time_val tv;
    tv.sec = (long)(us/SECS_TO_1_US);
    tv.usec = us%SECS_TO_1_US;

    return vos_settimeofday(&tv);
}

//ms
vos_uint32_t vos_get_system_tick()
{
    vos_uint64_t tick;
    tick = vos_get_system_usec() - g_system_start_time;

    return (vos_uint32_t)(tick/1000);

}

vos_uint32_t vos_get_system_tick_sec()
{
    return vos_get_system_tick() / 1000;
}
