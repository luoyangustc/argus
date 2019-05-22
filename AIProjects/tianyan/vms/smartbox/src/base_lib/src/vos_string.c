#define __STRING_C__

#include "vos_types.h"
#include "vos_assert.h"
#include "vos_string.h"
#include "vos_ctype.h"

char* vos_strdup(const char* str)
{
    vos_size_t len;
    char* copy;

    len = strlen(str) + 1;
    if (!(copy = (char*)VOS_MALLOC_BLK_T(char, len))) return 0;
    memcpy(copy, str, len);
    return copy;
}

vos_str_t vos_str(char *str)
{
    vos_str_t dst;
    dst.ptr = str;
    dst.slen = str ? vos_ansi_strlen(str) : 0;
    return dst;
}

char* vos_str2(const vos_str_t *src, char* dst, vos_ssize_t n)
{
    vos_assert(src && dst && (src->slen <= n - 1));

    memcpy(dst, src->ptr, src->slen);

    dst[src->slen] = '\0';

    return dst;
}

vos_str_t* vos_strncpy_with_null(vos_str_t *dst, const vos_str_t *src, vos_ssize_t max)
{
    if (max <= src->slen)
    {
        max = max - 1;
    }
    else
    {
        max = src->slen;
    }
    memcpy(dst->ptr, src->ptr, max);
    dst->ptr[max] = '\0';
    dst->slen = max;
    return dst;
}

#if defined(VOS_HAS_NO_SNPRINTF) && VOS_HAS_NO_SNPRINTF != 0
int snprintf(char *s1, vos_size_t len, const char *s2, ...)
{
    int ret;
    va_list arg;

    VOS_UNUSED_ARG(len);

    va_start(arg, s2);
    ret = vsprintf(s1, s2, arg);
    va_end(arg);
    
    return ret;
}

int vsnprintf(char *s1, vos_size_t len, const char *s2, va_list arg)
{
#define MARK_CHAR   ((char)255)
    int rc;

    s1[len-1] = MARK_CHAR;

    rc = vsprintf(s1,s2,arg);

    vos_assert(s1[len-1] == MARK_CHAR || s1[len-1] == '\0');

    return rc;
}
#endif


