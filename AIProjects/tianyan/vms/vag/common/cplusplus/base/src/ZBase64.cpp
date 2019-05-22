#include "ZBase64.h"
#include <string.h>

string ZBase64::Encode(const unsigned char* Data,int DataByte)
{
	//编码表
	const char EncodeTable[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	//返回值
	string strEncode;
	unsigned char Tmp[4]={0};
	for(int i=0;i<(int)(DataByte / 3);i++)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		Tmp[3] = *Data++;
		strEncode+= EncodeTable[Tmp[1] >> 2];
		strEncode+= EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
		strEncode+= EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
		strEncode+= EncodeTable[Tmp[3] & 0x3F];
	
	}
	//对剩余数据进行编码
	int Mod=DataByte % 3;
	if(Mod==1)
	{
		Tmp[1] = *Data++;
		strEncode+= EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode+= EncodeTable[((Tmp[1] & 0x03) << 4)];
		strEncode += "==";
	}
	else if(Mod==2)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		strEncode+= EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode+= EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
		strEncode+= EncodeTable[((Tmp[2] & 0x0F) << 2)];
		strEncode += "=";
	}

	return strEncode;
}

int ZBase64::Decode(const char* Data,int DataByte,unsigned char out_data[],int& OutByte)
{
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
	
	memset(out_data,0,OutByte);
	if(OutByte < DecodeLength(DataByte))
	{	    
		return 0;
	}

	int out_data_pos = 0;
	OutByte = DecodeLength(DataByte);
	int decode_pos = 0;
	unsigned char Tmp[4]={0};
	do
	{
		if (DataByte - decode_pos>=4)
		{
			Tmp[0] = DecodeTable[*Data++];
			Tmp[1] = DecodeTable[*Data++];
			Tmp[2] = DecodeTable[*Data++];
			Tmp[3] = DecodeTable[*Data++];

		
			out_data[out_data_pos++] = ((Tmp[0]<<2)&0xFC) | ((Tmp[1]>>4)&0X03);
			if( Tmp[2] == '@' )		
			{
			    out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) ;
				if (out_data[out_data_pos]==0)
				{
					--out_data_pos;
				}
			    break;
			}
			
			out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) | ((Tmp[2]>>2)&0X0f);

			if( Tmp[3] == '@')
			{
			    out_data[out_data_pos++] = ((Tmp[2]<<6)&0XC0);
				if (out_data[out_data_pos]==0)
				{
					--out_data_pos;
				}
			    break;
			}

			out_data[out_data_pos++] = ((Tmp[2]<<6)&0XC0) | (Tmp[3]&0X3F);
			decode_pos += 4;
		}
		else
		{
			break;
		}
	}while(true);
	OutByte = out_data_pos;
	return out_data_pos;
}

int ZBase64::EncodeLength(int src_length)
{
	int ret = (src_length+2)/3*4;
	return ret;
}

int ZBase64::DecodeLength(int src_length)
{
	int ret = (src_length+3)/4*3;
	return ret;
}

string ZBase64::Safe_Encode(const unsigned char* Data,int DataByte)
{
	//编码表
	const char EncodeTable[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
	//返回值
	string strEncode;
	unsigned char Tmp[4]={0};
	for(int i=0;i<(int)(DataByte / 3);i++)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		Tmp[3] = *Data++;
		strEncode+= EncodeTable[Tmp[1] >> 2];
		strEncode+= EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
		strEncode+= EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
		strEncode+= EncodeTable[Tmp[3] & 0x3F];

	}
	//对剩余数据进行编码
	int Mod=DataByte % 3;
	if(Mod==1)
	{
		Tmp[1] = *Data++;
		strEncode+= EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode+= EncodeTable[((Tmp[1] & 0x03) << 4)];
		//strEncode += "==";
	}
	else if(Mod==2)
	{
		Tmp[1] = *Data++;
		Tmp[2] = *Data++;
		strEncode+= EncodeTable[(Tmp[1] & 0xFC) >> 2];
		strEncode+= EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
		strEncode+= EncodeTable[((Tmp[2] & 0x0F) << 2)];
		//strEncode += "=";
	}

	return strEncode;
}

int ZBase64::Safe_Decode(const char* Data,int DataByte,unsigned char out_data[],int& OutByte)
{
	//解码表
	const char DecodeTable[] =
	{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          0, 0, 0, 0,//24
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0, 0, 0, 0, 0,//19
		0, // '+'
		0, 
		62,//- 
		0,
		0, // '/'
		52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
		0, 0, 0, 
		64,//=
		0,
		0, 0,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
		0, 0, 0, 0, 
		63, //_
		0,
		26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
	};

	memset(out_data,0,OutByte);
	if (Safe_DecodeLength(DataByte)<0)
	{
		return 0;
	}
	if(OutByte < Safe_DecodeLength(DataByte))
	{	    
		return 0;
	}

	int out_data_pos = 0;
	OutByte = Safe_DecodeLength(DataByte);
	int decode_pos = 0;
	unsigned char Tmp[4]={0};
	do
	{
		if (DataByte - decode_pos>=4)
		{
			Tmp[0] = DecodeTable[*Data++];
			Tmp[1] = DecodeTable[*Data++];
			Tmp[2] = DecodeTable[*Data++];
			Tmp[3] = DecodeTable[*Data++];

			out_data[out_data_pos++] = ((Tmp[0]<<2)&0xFC) | ((Tmp[1]>>4)&0X03);
			out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) | ((Tmp[2]>>2)&0X0f);
			out_data[out_data_pos++] = ((Tmp[2]<<6)&0XC0) | (Tmp[3]&0X3F);
			decode_pos += 4;
		}
		else if (DataByte - decode_pos==3)
		{
			Tmp[0] = DecodeTable[*Data++];
			Tmp[1] = DecodeTable[*Data++];
			Tmp[2] = DecodeTable[*Data++];

			out_data[out_data_pos++] = ((Tmp[0]<<2)&0xFC) | ((Tmp[1]>>4)&0X03);
			out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) | ((Tmp[2]>>2)&0X0f);

			{
				out_data[out_data_pos++] = ((Tmp[2]<<6)&0XC0);
				if (out_data[out_data_pos]==0)
				{
					--out_data_pos;
				}
			}
			break;
		}
		else if (DataByte - decode_pos==2)
		{
			Tmp[0] = DecodeTable[*Data++];
			Tmp[1] = DecodeTable[*Data++];

			out_data[out_data_pos++] = ((Tmp[0]<<2)&0xFC) | ((Tmp[1]>>4)&0X03);
			{
				out_data[out_data_pos++] = ((Tmp[1]<<4)&0xF0) ;
				if (out_data[out_data_pos]==0)
				{
					--out_data_pos;
				}
			}
			break;
		}
		else if (DataByte - decode_pos==1)
		{
			return 0;//解码失败
		}
		else
		{
			break;
		}
	}while(true);
	OutByte = out_data_pos;
	return out_data_pos;
}

int ZBase64::Safe_EncodeLength(int src_length)
{
	int ret = (src_length)/3*4;
	int mod = src_length%3;
	if (mod == 1)
	{
		ret += 2;
	}
	else if (mod == 2)
	{
		ret += 3;
	}
	return ret;
}

int ZBase64::Safe_DecodeLength(int src_length)
{
	int ret = (src_length)/4*3;
	int mod = src_length%4;
	if (mod == 2)
	{
		ret += 1;
	}
	else if (mod == 3)
	{
		ret += 2;
	}
	else if (mod == 1)
	{
		//如果出现到这里，说明解码长度不对
		return -1;
	}
	return ret;
}

