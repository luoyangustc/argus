#define __DNL_UTIL_C__

#include "dnl_def.h"
#include "dnl_util.h"
#include "dnl_log.h"


const char b64_alphabet[65] = { 
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"abcdefghijklmnopqrstuvwxyz"
	"0123456789+/=" };
 
//子函数 - 取密文的索引  
char GetCharIndex(char c) //内联函数可以省去函数调用过程，提速  
{   if((c >= 'A') && (c <= 'Z'))  
    {   
		return c - 'A';  
    }
    else if((c >= 'a') && (c <= 'z'))  
    {  
		return c - 'a' + 26;  
    }
	else if((c >= '0') && (c <= '9'))  
    {   
		return c - '0' + 52;  
    }
	else if(c == '+')  
    {  
		return 62;  
    }
	else if(c == '/')  
    {   
		return 63; 
	}
	else if(c == '=')  
	{   
		return 0;  
	}  
return 0;  
}  

//解码，参数：结果，密文，密文长度  
int fnBase64Decode(char *lpString, char *lpSrc, int sLen)   //解码函数  
{   
	static char lpCode[4];  
	register int vLen = 0;  
	if(sLen % 4)        //Base64编码长度必定是4的倍数，包括'='  
	{  
		lpString[0] = '\0';  
		return -1;  
	}  
	while(sLen > 2)      //不足三个字符，忽略  
	{   
		lpCode[0] = GetCharIndex(lpSrc[0]);  
		lpCode[1] = GetCharIndex(lpSrc[1]);  
		lpCode[2] = GetCharIndex(lpSrc[2]);  
		lpCode[3] = GetCharIndex(lpSrc[3]);  

		*lpString++ = (lpCode[0] << 2) | (lpCode[1] >> 4);  
		*lpString++ = (lpCode[1] << 4) | (lpCode[2] >> 2);  
		*lpString++ = (lpCode[2] << 6) | (lpCode[3]);  

		lpSrc += 4;  
		sLen -= 4;  
		vLen += 3;  
	}  
	return vLen;  
}  

int ZBase64DecodeLength(int src_length)
{
	int ret = src_length/4*3;
	return ret;
}


//解码表
	const char DecodeTable[] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 0, 0,//24
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0,//19
		62, // '+'
		0, 0, 0,
		63, // '/'
		52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
		0, 0, 0, 
		64,//=
		0,
		0, 0,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
		0, 0, 0, 0, 0, 0,
		26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
	};
	

int ZBase64Decode(const char* Data,int DataByte,unsigned char out_data[],int *OutByte)
{
	int out_data_pos = 0;
	int decode_pos = 0;
	unsigned char Tmp[4]={0};

	
	if (Data == NULL || OutByte == NULL)
	{
		return 0;
	}
	if(*OutByte < ZBase64DecodeLength(DataByte))
	{	    
		return 0;
	}

	*OutByte = ZBase64DecodeLength(DataByte);

	do
	{
		if (DataByte - decode_pos>=4)
		{
			Tmp[0] = DecodeTable[(int)(*Data++)];
			Tmp[1] = DecodeTable[(int)(*Data++)];
			Tmp[2] = DecodeTable[(int)(*Data++)];
			Tmp[3] = DecodeTable[(int)(*Data++)];

			out_data[out_data_pos++] = ((Tmp[0]<<2)&0xFC) | ((Tmp[1]>>4)&0X03);
			if( Tmp[2] == '@' )		
			{
				break;
			}

			out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) | ((Tmp[2]>>2)&0X0f);

			if( Tmp[3] == '@')
			{
				break;
			}

			out_data[out_data_pos++] = ((Tmp[2]<<6)&0XC0) | (Tmp[3]&0X3F);
			decode_pos += 4;
		}
		else
		{
			break;
		}
	}while(1);
	return out_data_pos;
}

/*
* C=A^B
* A=B^C
*/
//算法1的加密实现
int ay_udp_msg_encrypt(void *databuf, int datalen)
{
	int 			i = 0;
	int 			start_key_pos = 0;
	unsigned char 	key = 0;
	unsigned char 	algo = 1;
	unsigned char *pbuf;
	
	pbuf = (unsigned char *)databuf;
	do 
	{
		if( !databuf )
		{
			return -1;
		}

		//产生随机key
		//srand((int)time(NULL));
		key = rand()%256;
		start_key_pos = DEVICE_1_KEY_POS;

		if(datalen < start_key_pos)
		{
			break;
		}

		//保存在flag的最高字节
		pbuf[start_key_pos] = key;

		if(datalen < start_key_pos)
		{
			break;
		}

		key = key|0x01;

		for(i = 0; i<datalen; ++i)
		{
			if( i != 1 && i != start_key_pos)
			{
				pbuf[i] = pbuf[i]^key;
				key = pbuf[i];
			}
		}

		//加密成功后再置算法位
		algo <<= 3;
		pbuf[1] |= algo;
	
		return 0;
	} while (0);
	return -1;
}

//算法1的解密实现
int ay_udp_msg_decrypt(void *databuf, int datalen)
{
	unsigned char testkey;
	unsigned char get_algo;
	unsigned char key;
	unsigned char *pbuf;

	int start_key_pos = 0;
	int i = 0;
	
	pbuf = (unsigned char *)databuf;

	do 
	{
		if( !pbuf )
		{
			return -1;
		}

		get_algo = pbuf[1];
		get_algo >>= 3;

		if (get_algo == 1)
		{
			//设备ID长度所在的位置
			start_key_pos = DEVICE_1_KEY_POS;

			if(datalen < start_key_pos)
			{
				break;
			}

			//选定设备ID的最后一个字节为KEY
			key = pbuf[start_key_pos];
			key = key|0x01;

			for( i = 0; i<datalen; ++i)
			{
				if( i != 1 && i != start_key_pos)
				{
					testkey = pbuf[i];
					pbuf[i] = pbuf[i]^key;
					key = testkey;
				}
			}

			pbuf[1] &= 0x07;//((~get_algo)|0x07);//0x7F;
			pbuf[start_key_pos] &=0x0;
		}
		return 0;
	} while (0);
	return -1;
}

int ay_tcp_msg_encrypt(void *databuf, int datalen, void *keybuf, int keylen)
{
	int i = 0;
	int start_key_pos = 0;
	int encry_len = 0;
	unsigned char key = 0;
	unsigned char *pbuf, *pkey;

	pbuf = (unsigned char *)databuf;
	pkey = (unsigned char *)keybuf;

	do 
	{
		{
			//产生随机key
			if(pbuf == NULL || pkey == NULL)
			{
				break;
			}
			if(datalen > 128)
			{
				encry_len = 128;
			}
			else
			{
				encry_len = datalen;
			}

			//产生随机key
			//srand((int)time(NULL));
			key = rand()%256;

			start_key_pos = DEVICE_1_KEY_POS;

			if(datalen < start_key_pos)
			{
				break;
			}

			//保存在flag的最高字节
			pbuf[start_key_pos] = key;

			key = key|0x01;
			
			for(i = 0; i < keylen; i++)
			{
				pkey[i] = pkey[i]^key;
				key = pkey[i];
			}

			for(i = 2; i < encry_len; ++i)
			{
				if(i != start_key_pos)
				{
				   pbuf[i] = pbuf[i]^pkey[i%keylen];
				}
			}

			//加密成功后再置算法位
			//algo <<= 3;
			//buf[1] |= algo;
		}

		return 0;

	} while (0);

	return -1;
}

int ay_tcp_msg_decrypt(void *databuf, int datalen, void *keybuf, int keylen)
{
	unsigned char key;
	unsigned char *pbuf, *pkey; /*, *pencry_key;*/
	int i, encry_len, start_key_pos;

	pbuf = (unsigned char*)databuf;
	pkey = (unsigned char*)keybuf;

	do 
	{
		if (pbuf == NULL || pkey == NULL)
		{
			break;
		}

		//设备ID长度所在的位置
		start_key_pos = DEVICE_1_KEY_POS;
		if(datalen < start_key_pos)
		{
			break;
		}
		//memcpy(&encry_key[0],pkey,key_len);
		if (datalen > 128)
		{
			encry_len = 128;
		}
		else
		{
			encry_len = datalen;
		}
		//pencry_key = pkey;


		//选定设备ID的最后一个字节为KEY
		key = pbuf[start_key_pos];
		key = key|0x01;


		for(i = 0; i<keylen; i++)
		{
			pkey[i] = pkey[i]^key;
			key = pkey[i];
		}

		//pencry_key = pkey;
		for(i = 2; i < encry_len; ++i)
		{
			if(i != start_key_pos)
			{
				pbuf[i] = pbuf[i]^pkey[i%keylen];
				//key = buf[i];
			}	 
		}

		pbuf[start_key_pos] &=0x0;

		return 0;

	} while (0);

	return -1;
}


static void print_error(char *s)
{

	fputs(s, stderr); 
	fputc('\n', stderr);
	//exit(1);
}

void status(ghttp_request *r, char *desc)
{

	ghttp_current_status st;

	st = ghttp_get_status(r);

	//	fprintf(stderr, "%s: %s [%d/%d]\n",

	//		desc,

	//		st.proc == ghttp_proc_request ? "request" :

	//		st.proc == ghttp_proc_response_hdrs ? "response-headers" :

	//		st.proc == ghttp_proc_response ? "response" : "none",

	//		st.bytes_read, st.bytes_total);

}

ghttp_request *request_webserver_new(void)
{
	return ghttp_request_new();
}

void request_webserver_destroy(ghttp_request *pReq)
{
	ghttp_request_destroy(pReq);
	return;
}

char *request_webserver_content(const char *purl,ghttp_request *pReq,int *pout_len)
{
	int bytes = 0;
	ghttp_status req_status;

	if (purl == NULL || pReq == NULL)
	{
		return NULL;
	}

	DNL_DEBUG_LOG("request_webserver_content, url=%s\n", purl);

	if (ghttp_set_uri(pReq,(char*)purl) < 0)
	{
		print_error("ghttp_set_uri");
		return NULL;
	}
	DNL_DEBUG_LOG("set uri ok.");

	if (ghttp_prepare(pReq) < 0)
	{
		print_error("ghttp_prepare");
		return NULL;
	}
	DNL_DEBUG_LOG("set ghttp_prepare ok.");

	if (ghttp_set_sync(pReq, ghttp_async) < 0)
	{
		//print_error("ghttp_set_sync");
		return NULL;
	}
	DNL_DEBUG_LOG("set ghttp_set_sync ok.");

	do {

		status(pReq, "conn0");
		req_status = ghttp_process(pReq);
		if (req_status == ghttp_error)
		{
			fprintf(stderr, "ghttp err: %s\n",ghttp_get_error(pReq));
			return NULL;
		}

		if (req_status != ghttp_error && ghttp_get_body_len(pReq) > 0) 
		{
			bytes += ghttp_get_body_len(pReq);
		}
	} while (req_status == ghttp_not_done);

	if (pout_len)
	{
		*pout_len = ghttp_get_body_len(pReq);
	}

	return ghttp_get_body(pReq);
}

int get_json_len_byfile(const char* in_path)
{
#if 0
	int file_len = 0;
	FILE* fp = NULL;
	fp = fopen(in_path, "rb");
	if ( !fp )
	{
		return -1;
	}

	if(fread(&file_len,sizeof(int),1,fp)!=1)
	{
		fclose(fp);
		return -1;
	}
	DNL_DEBUG_LOG("file_len: %d",file_len);
	
	fclose(fp);
	return file_len;
#else
    return -1;
#endif

}

int get_json_byfile(const char* in_path,char* ou_pContent,int buf_len)

{
#if 0
	char *pPos = NULL;
	int file_len = 0;
	FILE* fp = NULL;
	if (ou_pContent == NULL || in_path == NULL)
	{
		return -1;
	}
	fp = fopen(in_path, "rb");
	if ( !fp )
	{
		return -1;
	}

	if(fread(&file_len,sizeof(int),1,fp)!=1)
	{
		fclose(fp);
		return -1;
	}

	DNL_DEBUG_LOG("file_len: %d",file_len);

	if (file_len > buf_len)
	{
		fclose(fp);
		return -1;
	}
	pPos = ou_pContent;

	while(fread(pPos,1,1,fp)==1)
	{
		pPos++;
		if (pPos == NULL)
		{
			break;
		}
	}
	fclose(fp);
	return file_len;
#else
    return -1;
#endif
}

int write_json_byfile(const char* in_path,char* in_pContent,int content_len)
{
#if 0
	FILE* fp = fopen(in_path, "w");
	if (fp != 0)
	{
		fwrite(&content_len,1,sizeof(int),fp);
		fwrite(in_pContent, 1, content_len, fp);
		fclose(fp);
	}
#endif
	return 0;
}

int json_info_decode(const char* pin_Content,int in_len, char* pou_Content,int buf_len)
{
	char *pb64_buf = NULL;
	//int decode_len = 0;
	int decode_data_len = 0;
	pb64_buf = VOS_MALLOC_BLK_T(char, in_len+1);//(char*)malloc(in_len);
	if (!pb64_buf)
	{
		return -1;
	}

	decode_data_len = in_len;
	memset(pb64_buf,0,in_len+1);
#if 0
	if (hex_decode((char*)pin_Content, (unsigned char*)pb64_buf, &decode_data_len) < 0)
	{
		VOS_FREE_T(pb64_buf);
		return -1;
	}

	if(DES_Decrypt((unsigned char*)pb64_buf, decode_data_len, (unsigned char *)ANYAN_DEFAULT_KEY, 8,(unsigned char*)pou_Content, buf_len,&decode_data_len) != ANYAN_DES_OK)
	{
		VOS_FREE_T(pb64_buf);
		return -1;
	}
#else
    memcpy(pou_Content, pin_Content, in_len);
#endif
	VOS_FREE_T(pb64_buf);
	return decode_data_len;
}




