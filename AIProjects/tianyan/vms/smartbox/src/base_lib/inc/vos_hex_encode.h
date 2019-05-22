#ifndef __VOS_UTIL_HEX_ENCODE_H__
#define __VOS_UTIL_HEX_ENCODE_H__

#include "vos_types.h"

#undef EXT
#ifndef __UTIL_HEX_ENCODE_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

EXT int hex_encode(unsigned char * buff,int buff_len,char out_buff[]);
EXT int hex_decode(char buff[],unsigned char out_buff[],int * out_buff_len);

VOS_END_DECL

#endif //__VOS_UTIL_HEX_ENCODE_H__

