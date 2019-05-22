#ifndef __VOS_UTIL_BASE64_H__
#define __VOS_UTIL_BASE64_H__

#include "vos_types.h"

#undef EXT
#ifndef __UTIL_BASE64_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL


#define BASE256_TO_BASE64_LEN(len)	(len * 4 / 3 + 3)

#define BASE64_TO_BASE256_LEN(len)	(len * 3 / 4)


EXT vos_status_t base64_encode(const vos_uint8_t *input, int in_len,
				     char *output, int *out_len);

EXT vos_status_t base64_decode(const char *input, 
				      vos_uint8_t *out, int *out_len);


VOS_END_DECL


#endif		//__VOS_UTIL_BASE64_H__


