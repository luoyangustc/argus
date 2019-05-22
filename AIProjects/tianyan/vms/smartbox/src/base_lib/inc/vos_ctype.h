#ifndef __VOS_CTYPE_H__
#define __VOS_CTYPE_H__

#include "vos_types.h"

#if (VOS_HAS_CTYPE_H == 1)
#include <ctype.h>
#else
#  define isalnum(c)	    (isalpha(c) || isdigit(c))
#  define isalpha(c)	    (islower(c) || isupper(c))
#  define isascii(c)	    (((unsigned char)(c))<=0x7f)
#  define isdigit(c)	    ((c)>='0' && (c)<='9')
#  define isspace(c)	    ((c)==' ' || (c)=='\t' || (c)=='\n' || (c)=='\r' || (c)=='\v')
#  define islower(c)	    ((c)>='a' && (c)<='z')
#  define isupper(c)	    ((c)>='A' && (c)<='Z')
#  define isxdigit(c)	    (isdigit(c) || (tolower(c)>='a'&&tolower(c)<='f'))
#  define tolower(c)	    (((c) >= 'A' && (c) <= 'Z') ? (c)+('a'-'A') : (c))
#  define toupper(c)	    (((c) >= 'a' && (c) <= 'z') ? (c)-('a'-'A') : (c))
#endif

#ifndef isblank
#   define isblank(c)	    (c==' ' || c=='\t')
#endif

VOS_BEGIN_DECL

#define vos_hex_digits	"0123456789abcdef"

VOS_INLINE_SPECIFIER int vos_isalnum(unsigned char c) { return isalnum(c); }
VOS_INLINE_SPECIFIER int vos_isalpha(unsigned char c) { return isalpha(c); }
VOS_INLINE_SPECIFIER int  vos_isascii(unsigned char c) { return c<128; }
VOS_INLINE_SPECIFIER int  vos_isdigit(unsigned char c) { return isdigit(c); }
VOS_INLINE_SPECIFIER int  vos_isspace(unsigned char c) { return isspace(c); }
VOS_INLINE_SPECIFIER int  vos_islower(unsigned char c) { return islower(c); }
VOS_INLINE_SPECIFIER int  vos_isupper(unsigned char c) { return isupper(c); }
VOS_INLINE_SPECIFIER int  vos_isblank(unsigned char c) { return isblank(c); }
VOS_INLINE_SPECIFIER int  vos_tolower(unsigned char c) { return tolower(c); }
VOS_INLINE_SPECIFIER int  vos_toupper(unsigned char c) { return toupper(c); }
VOS_INLINE_SPECIFIER int  vos_isxdigit(unsigned char c){ return isxdigit(c); }

VOS_INLINE_SPECIFIER void vos_val_to_hex_digit(unsigned value, char *p)
{
    *p++ = vos_hex_digits[ (value & 0xF0) >> 4 ];
    *p   = vos_hex_digits[ (value & 0x0F) ];
}
VOS_INLINE_SPECIFIER unsigned vos_hex_digit_to_val(unsigned char c)
{
    if (c <= '9')
	return (c-'0') & 0x0F;
    else if (c <= 'F')
	return  (c-'A'+10) & 0x0F;
    else
	return (c-'a'+10) & 0x0F;
}

VOS_END_DECL

#endif	/* __VOS_CTYPE_H__ */


