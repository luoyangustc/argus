
#ifndef __NETCRYPT_H__
#define __NETCRYPT_H__

#include <stdio.h>
class CAYCrypt
{
public:
	CAYCrypt();
	~CAYCrypt();
public:
	static int EncryUdpMsg(char* buff, int buff_len);
	static int DecryUdpMsg(char* buff, int buff_len);
	static int EncryTcpMsg(unsigned char* buf, int buf_len, int key_pos, unsigned char *pkey, int key_len);
	static int DecryTcpMsg(unsigned char* buf, int buf_len, int key_pos, unsigned char *pkey, int key_len);
};

#endif /* __NETCRYPT_H__ */
