
#ifndef __VOS_UNICODE_H__
#define __VOS_UNICODE_H__

#include "vos_types.h"

VOS_BEGIN_DECL

wchar_t* vos_ansi_to_unicode(const char *str, vos_size_t len,
				     wchar_t *wbuf, vos_size_t wbuf_count);

char* vos_unicode_to_ansi(const wchar_t *wstr, vos_size_t len,
				  char *buf, vos_size_t buf_size);


#if defined(_UNICODE) && _UNICODE!=0
#   define VOS_DECL_UNICODE_TEMP_BUF(buf,size)   wchar_t buf[size];
#   define VOS_STRING_TO_NATIVE(s,buf,max)	vos_ansi_to_unicode( \
						    s, strlen(s), \
						    buf, max)
#   define VOS_DECL_ANSI_TEMP_BUF(buf,size)	char buf[size];
#   define VOS_NATIVE_TO_STRING(cs,buf,max)	vos_unicode_to_ansi( \
						    cs, wcslen(cs), \
						    buf, max)
#else
#   define VOS_DECL_UNICODE_TEMP_BUF(var,size) char var[size];
#   define VOS_STRING_TO_NATIVE(s,buf,max)	((char*)s)
#   define VOS_DECL_ANSI_TEMP_BUF(buf,size)
#   define VOS_NATIVE_TO_STRING(cs,buf,max)	((char*)(const char*)cs)
#endif

VOS_END_DECL


#endif	/* __VOS_UNICODE_H__ */

