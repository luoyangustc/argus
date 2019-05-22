#ifndef __BIT_T_H__
#define __BIT_T_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char bit1;
typedef unsigned char bit2;
typedef unsigned char bit3;
typedef unsigned char bit4;
typedef unsigned char bit5;
typedef unsigned char bit6;
typedef unsigned char bit7;
typedef unsigned char bit8;
typedef unsigned short bit9;
typedef unsigned short bit10;
typedef unsigned short bit11;
typedef unsigned short bit12;
typedef unsigned short bit13;
typedef unsigned short bit14;
typedef unsigned short bit15;
typedef unsigned short bit16;
typedef unsigned int bit17;
typedef unsigned int bit18;
typedef unsigned int bit19;
typedef unsigned int bit20;
typedef unsigned int bit21;
typedef unsigned int bit22;
typedef unsigned int bit23;
typedef unsigned int bit24;
typedef unsigned int bit25;
typedef unsigned int bit26;
typedef unsigned int bit27;
typedef unsigned int bit28;
typedef unsigned int bit29;
typedef unsigned int bit30;
typedef unsigned int bit31;
typedef unsigned int bit32;
//typedef vos_uint64_t bit33;
//typedef vos_uint64_t bit34;
//typedef vos_uint64_t bit35;
//typedef vos_uint64_t bit36;
//typedef vos_uint64_t bit40;
//typedef vos_uint64_t bit64;

#define R1B7(d) (d >> 7)
#define R1B6(d) ((d >> 6) & 0x01)
#define R1B5(d) ((d >> 5) & 0x01)
#define R1B4(d) ((d >> 4) & 0x01)
#define R1B3(d) ((d >> 3) & 0x01)
#define R1B2(d) ((d >> 2) & 0x01)
#define R1B1(d) ((d >> 1) & 0x01)
#define R1B0(d) (d & 0x01)

#define R2B7(d) (d >> 6)
#define R2B6(d) ((d >> 5) & 0x03)
#define R2B5(d) ((d >> 4) & 0x03)
#define R2B4(d) ((d >> 3) & 0x03)
#define R2B3(d) ((d >> 2) & 0x03)
#define R2B2(d) ((d >> 1) & 0x03)
#define R2B1(d) (d & 0x03)

#define R3B7(d) (d >> 5)
#define R3B6(d) ((d >> 4) & 0x07)
#define R3B5(d) ((d >> 3) & 0x07)
#define R3B4(d) ((d >> 2) & 0x07)
#define R3B3(d) ((d >> 1) & 0x07)
#define R3B2(d) (d & 0x07)

#define R4B7(d) (d >> 4)
#define R4B6(d) ((d >> 3) & 0x0f)
#define R4B5(d) ((d >> 2) & 0x0f)
#define R4B4(d) ((d >> 1) & 0x0f)
#define R4B3(d) (d & 0x0f)

#define R5B7(d) (d >> 3)
#define R5B6(d) ((d >> 2) & 0x1f)
#define R5B5(d) ((d >> 1) & 0x1f)
#define R5B4(d) (d & 0x1f)

#define R6B7(d) (d >> 2)
#define R6B6(d) ((d >> 1) & 0x3f)
#define R6B5(d) (d & 0x3f)

#define R7B7(d) (d >> 1)
#define R7B6(d) (d & 0x7f)

#define R8BITS(p, d) \
	*(unsigned char*)&d = *p++

#define R16BITS(p, d) \
    *(unsigned char*)&d = *p++;\
	*(((unsigned char*)&d) + 1) = *p++
	

#define R24BITS(p, d) \
    *(unsigned char*)&d = *p++;\
	*(((unsigned char*)&d) + 1) = *p++;\
	*(((unsigned char*)&d) + 2) = *p++;\
	*(((unsigned char*)&d) + 3) = 0
	
#define R32BITS(p, d) \
    *(unsigned char*)&d = *p++;\
	*(((unsigned char*)&d) + 1) = *p++;\
	*(((unsigned char*)&d) + 2) = *p++;\
	*(((unsigned char*)&d) + 3) = *p++


#define SET8BITS(p, d) \
	*p++ = *(unsigned char*)&d 

#define SET16BITS(p, d) \
    *p++ = *(unsigned char*)&d;\
	*p++ = *(((unsigned char*)&d) + 1)

#define SET32BITS(p, d) \
    *p++ = *(unsigned char*)&d ;\
	*p++ = *(((unsigned char*)&d) + 1) ;\
	*p++ = *(((unsigned char*)&d) + 2) ;\
	*p++ = *(((unsigned char*)&d) + 3) 

void MemeryCopy(void *src, void *dst, int n)
{
	int i = 0;
	unsigned char *pdst = (unsigned char*)dst;
	unsigned char *psrc = (unsigned char*)src;

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


#define ForwardBytes(p, n)   (p +=n) /*forward n bytes*/

#define R1Bytes(p, d) R8BITS(p, d) 
#define R2Bytes(p, d) R16BITS(p, d) 
#define R3Bytes(p, d) R24BITS(p, d) 
#define R4Bytes(p, d) R32BITS(p, d) 
#define RnBytes(p, d, n) do{MemeryCopy(p,d,n);p += n;}while(0)
#define RnString(p, d) do{ p += StringCopy(p,d); }while(0)

#define S1Bytes(p, d) SET8BITS(p, d) 
#define S2Bytes(p, d) SET16BITS(p, d) 
#define S4Bytes(p, d) SET32BITS(p, d)
#define SnBytes(p, d, n) do{MemeryCopy(d,p,n);p += n;}while(0)
#define SnString(p, d) do{ p += StringCopy(d, p);}while(0)

#endif			//__BIT_T_H__
