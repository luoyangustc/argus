#ifndef __DNL_UTIL_H__
#define __DNL_UTIL_H__

#include "comm_includes.h"

#undef  EXT
#ifndef __DNL_UTIL_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL

EXT char GetCharIndex(char c);
EXT int fnBase64Decode(char *lpString, char *lpSrc, int sLen);
EXT int ZBase64DecodeLength(int src_length);
EXT int ZBase64Decode(const char* Data,int DataByte,unsigned char out_data[],int *OutByte);

EXT int ay_udp_msg_encrypt(void *databuf, int datalen);
EXT int ay_udp_msg_decrypt(void *databuf, int datalen);
EXT int ay_tcp_msg_encrypt(void *databuf, int datalen, void *keybuf, int keylen);
EXT int ay_tcp_msg_decrypt(void *databuf, int datalen, void *keybuf, int keylen);

EXT int get_json_byfile(const char* in_path,char* ou_pContent,int buf_len);
EXT int write_json_byfile(const char* in_path,char* in_pContent,int content_len);
EXT int get_json_len_byfile(const char* in_path);
EXT int json_info_decode(const char* pin_Content,int in_len, char* pou_Content,int buf_len);
EXT ghttp_request *request_webserver_new(void);
EXT void request_webserver_destroy(ghttp_request *pReq);
EXT char *request_webserver_content(const char *purl,ghttp_request *pReq,int *pout_len);

VOS_END_DECL

#endif