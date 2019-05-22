#define __UTIL_HEX_ENCODE_C__
#include "vos_hex_encode.h"
#include <stdio.h>
#include "vos_string.h"

int hex_encode(unsigned char * buff,int buff_len,char out_buff[])
{
    int i = 0;
    
    if(buff_len<=0 || buff_len > 1024*16)
    {
		return -1;
    }
    out_buff[buff_len*2] = 0;
    for(; i<buff_len; ++i)
    {
		sprintf(out_buff+2*i,"%02x",buff[i]);
    }
    return 0;
}

unsigned char get_hex_value(char c)
{
	unsigned char ret = 255;
	if ( c >='0'&& c <='9')
	{
		ret = c - '0';
	}
	else if (c>='A'&& c<='F')
	{
		ret = c - 'A' + 10;
	}
	else if (c>='a'&& c<='f')
	{
		ret = c - 'a' + 10;
	}
	
	return ret;
}

int hex_decode(char buff[],unsigned char out_buff[],int * out_buff_len)
{
    int i = 0;
	int buff_len = strlen(buff);
	if (buff_len<=0 || buff_len%2 || buff_len > 1024*16)
	{
		return -1;
	}

	*out_buff_len = buff_len/2;
	for (i = 0; i<*out_buff_len; ++i)
	{
		unsigned char tmp1 = get_hex_value(buff[2*i]);
		unsigned char tmp2 = get_hex_value(buff[2*i+1]);
		
		if (tmp1 == 255 || tmp2 == 255)
		{
			return -1;
		}
		
		out_buff[i] = tmp1*16+tmp2;
	}
    return 0;
}

