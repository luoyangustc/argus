
#ifndef __VOS_STRING_H__
#define __VOS_STRING_H__

#undef EXT
#ifndef __STRING_C__
#define EXT    extern
#else
#define EXT
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vos_types.h"

#if defined(_MSC_VER)
#   define strcasecmp		_stricmp
#   define strncasecmp		_strnicmp
#   define snprintf			_snprintf
#   define vsnprintf			_vsnprintf
#   define snwprintf			_snwprintf
#   define vsnwprintf			_vsnwprintf
#   define wcsicmp			_wcsicmp
#   define wcsnicmp			_wcsnicmp
#else
#   define stricmp	strcasecmp
#   define strnicmp	strncasecmp

#   if defined(_UNICODE) && _UNICODE!=0
#	error "Implement Unicode string functions"
#   endif
#endif

#define vos_ansi_strcmp			strcmp
#define vos_ansi_strncmp			strncmp
#define vos_ansi_strlen				strlen
#define vos_ansi_strcpy			strcpy
#define vos_ansi_strncpy			strncpy
#define vos_ansi_strcat			strcat
#define vos_ansi_strstr				strstr
#define vos_ansi_strchr				strchr
#define vos_ansi_strcasecmp		strcasecmp
#define vos_ansi_stricmp			strcasecmp
#define vos_ansi_strncasecmp	strncasecmp
#define vos_ansi_strnicmp			strncasecmp
#define vos_ansi_sprintf			sprintf


#define vos_ansi_snprintf		snprintf
#define vos_ansi_vsprintf		vsprintf
#define vos_ansi_vsnprintf		vsnprintf

#define vos_unicode_strcmp		wcscmp
#define vos_unicode_strncmp		wcsncmp
#define vos_unicode_strlen		wcslen
#define vos_unicode_strcpy		wcscpy
#define	vos_unicode_strncpy	wcsncpy
#define vos_unicode_strcat		wcscat
#define vos_unicode_strstr		wcsstr
#define vos_unicode_strchr		wcschr
#define vos_unicode_strcasecmp	wcsicmp
#define vos_unicode_stricmp			wcsicmp
#define vos_unicode_strncasecmp	wcsnicmp
#define vos_unicode_strnicmp		wcsnicmp
#define vos_unicode_sprintf			swprintf
#define vos_unicode_snprintf			snwprintf
#define vos_unicode_vsprintf			vswprintf
#define vos_unicode_vsnprintf		vsnwprintf

#if defined(_UNICODE) && _UNICODE!=0
#   define vos_native_strcmp			vos_unicode_strcmp
#   define vos_native_strncmp		vos_unicode_strncmp
#   define vos_native_strlen			vos_unicode_strlen
#   define vos_native_strcpy			vos_unicode_strcpy
#   define vos_native_strncpy	    vos_unicode_strncpy
#   define vos_native_strcat			vos_unicode_strcat
#   define vos_native_strstr			vos_unicode_strstr
#   define vos_native_strchr			vos_unicode_strchr
#   define vos_native_strcasecmp	vos_unicode_strcasecmp
#   define vos_native_stricmp			vos_unicode_stricmp
#   define vos_native_strncasecmp	vos_unicode_strncasecmp
#   define vos_native_strnicmp		vos_unicode_strnicmp
#   define vos_native_sprintf			vos_unicode_sprintf
#   define vos_native_snprintf	    vos_unicode_snprintf
#   define vos_native_vsprintf	    vos_unicode_vsprintf
#   define vos_native_vsnprintf	    vos_unicode_vsnprintf
#else
#   define vos_native_strcmp			vos_ansi_strcmp
#   define vos_native_strncmp		vos_ansi_strncmp
#   define vos_native_strlen			vos_ansi_strlen
#   define vos_native_strcpy			vos_ansi_strcpy
#   define vos_native_strncpy	    vos_ansi_strncpy
#   define vos_native_strcat			vos_ansi_strcat
#   define vos_native_strstr			vos_ansi_strstr
#   define vos_native_strchr			vos_ansi_strchr
#   define vos_native_strcasecmp	vos_ansi_strcasecmp
#   define vos_native_stricmp			vos_ansi_stricmp
#   define vos_native_strncasecmp	vos_ansi_strncasecmp
#   define vos_native_strnicmp		vos_ansi_strnicmp
#   define vos_native_sprintf			vos_ansi_sprintf
#   define vos_native_snprintf	    vos_ansi_snprintf
#   define vos_native_vsprintf	    vos_ansi_vsprintf
#   define vos_native_vsnprintf	    vos_ansi_vsnprintf
#endif

#define vos_bzero(dst, size)    memset(dst, 0, size)

struct vos_str_t
{
    char								*ptr;
    vos_ssize_t							slen;
};
EXT char* vos_strdup(const char* str);
EXT vos_str_t vos_str(char *str);
EXT char* vos_str2(const vos_str_t *src, char* dst, vos_ssize_t n);
EXT vos_str_t* vos_strncpy_with_null(vos_str_t *dst, const vos_str_t *src, vos_ssize_t max);

#endif /*__VOS_STRING_H__*/

