#ifndef  __DNL_DEF_H__
#define __DNL_DEF_H__

#include "protocol_header.h"

#define ANYAN_DES_CIPHER_ERROR      -1 
#define ANYAN_DES_OK                0  
#define ANYAN_DEFAULT_KEY           "12345678"

#define DEVICE_MAX_LEN      20
#define ADDR_MAX_LEN        16
#define TOKEN_MAX_LEN       384
#define PATH_MAX_LEN        32  //255

#define STREAM_ENCRYPT      0   //媒体流是否加密0:不加密， 1:加密
#define DEBUG_FLAG          0   // 0 关闭debug  1 开启debug

typedef struct addr_info
{
    char IP[MAX_IP_LEN+1];
    vos_uint32_t sin_addr;
    vos_uint32_t port;
}addr_info;

#endif	/*__DNL_DEF_H__*/
