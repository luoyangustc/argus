#define __BS_TYPE_C__

#include "vos_bit_t.h"

void MemeryCopy(void *src, void *dst, int n)
{
	int i = 0;
	vos_uint8_t *pdst = (vos_uint8_t*)dst;
	vos_uint8_t *psrc = (vos_uint8_t*)src;

	for (; i < n; i++)
	{
		*pdst++ = *psrc++;
	}
}

int StringCopy(void *src, void *dst)
{
	char *pdst = (char*)dst;
	char *pos = (char*)src;
	char *psrc = (char*)src;

	
	do
	{
		*pdst++= *pos;
	}while(*pos++);

	return (pos - psrc);
}

#if 0
void R12B(uint8_t*& p, uint16_t& d)
{
	d = 0x0f & *p++;
	d <<= 8;
	d |= *p++;
}

void R13B(uint8_t*& p, uint16_t& d)
{
	d = 0x1f & *p++;
	d <<= 8;
	d |= *p++;
}

void R22B(uint8_t*& p, uint32_t& d)
{
	d = R7B6(*p++);
	d <<= 8;
	d |= *p++;
	d <<= 7;
	d |= R7B7(*p++) >> 1;
}

void R33B(uint8_t*& p, uint64_t& ts)
{
	ts = *p++;
	ts &= 0x0e;
	ts <<= 7;
	ts |= *p++;
	ts <<= 8;
	ts |= (*p++) & 0xfe;
	ts <<= 7;
	ts |= *p++;
	ts <<= 7;
	ts |= ((*p++) & 0xfe) >> 1;
}

void R33n10B(uint8_t*& p, uint64_t& base, uint16_t& ext)
{
	base = R3B5(*p); 
	base |= R2B1(*p++) << 1;
	base <<= 7;
	base |= *p++;
	base <<= 8;
	base |= R5B7(*p);
	base <<= R2B1(*p++) << 1;
	base <<= 7;
	base |= *p++;
	base <<= 5;
	base |= R5B7(*p) >> 3;
	ext = R2B1(*p++);
	ext <<= 7;
	ext |= R7B7(*p++) >> 1;
}

void RBytes(uint8_t*& p, uint32_t& data, int n)
{
	data = 0;
	for (int i = 0; i < n; i++)
	{
		data <<= 8;
		data += *p++;
	}
}
#endif //#if 0x01

