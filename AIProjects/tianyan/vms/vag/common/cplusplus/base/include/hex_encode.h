#ifndef __HEX_ENCODE_H__
#define __HEX_ENCODE_H__

int hex_encode(unsigned char * buff,int buff_len,char out_buff[]);
int hex_decode(char buff[],unsigned char out_buff[],int * out_buff_len);


#endif //__HEX_ENCODE_H__

