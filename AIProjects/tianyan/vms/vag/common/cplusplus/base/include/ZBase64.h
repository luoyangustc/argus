#ifndef __ZBASE64_H__
#define __ZBASE64_H__

//以下代码摘自http://www.cnblogs.com/phinecos/archive/2008/10/10/1308272.html
#include <string>
using namespace std;

class ZBase64
{
public:
    static string Encode(const unsigned char* Data,int DataByte);
    static int EncodeLength(int src_length);
    static int Decode(const char* Data,int DataByte,unsigned char out_data[],int& OutByte);
    static int DecodeLength(int src_length);

	static string Safe_Encode(const unsigned char* Data,int DataByte);
	static int Safe_EncodeLength(int src_length);
	static int Safe_Decode(const char* Data,int DataByte,unsigned char out_data[],int& OutByte);
	static int Safe_DecodeLength(int src_length);
};

#endif //__ZBASE64_H__

