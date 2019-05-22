#ifndef __VOS_BS_TYPE_H__
#define __VOS_BS_TYPE_H__

#include "vos_types.h"
#include "vos_string.h"

#undef EXT
#ifndef __BS_TYPE_C__
#define EXT extern
#else
#define EXT
#endif

VOS_BEGIN_DECL


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
typedef vos_uint32_t bit17;
typedef vos_uint32_t bit18;
typedef vos_uint32_t bit19;
typedef vos_uint32_t bit20;
typedef vos_uint32_t bit21;
typedef vos_uint32_t bit22;
typedef vos_uint32_t bit23;
typedef vos_uint32_t bit24;
typedef vos_uint32_t bit25;
typedef vos_uint32_t bit26;
typedef vos_uint32_t bit27;
typedef vos_uint32_t bit28;
typedef vos_uint32_t bit29;
typedef vos_uint32_t bit30;
typedef vos_uint32_t bit31;
typedef vos_uint32_t bit32;
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
	*(vos_uint8_t*)&d = *p++

#define R16BITS(p, d) \
    *(vos_uint8_t*)&d = *p++;\
	*(((vos_uint8_t*)&d) + 1) = *p++
	

#define R24BITS(p, d) \
    *(vos_uint8_t*)&d = *p++;\
	*(((vos_uint8_t*)&d) + 1) = *p++;\
	*(((vos_uint8_t*)&d) + 2) = *p++;\
	*(((vos_uint8_t*)&d) + 3) = 0
	
#define R32BITS(p, d) \
    *(vos_uint8_t*)&d = *p++;\
	*(((vos_uint8_t*)&d) + 1) = *p++;\
	*(((vos_uint8_t*)&d) + 2) = *p++;\
	*(((vos_uint8_t*)&d) + 3) = *p++


#define SET8BITS(p, d) \
	*p++ = *(vos_uint8_t*)&d 

#define SET16BITS(p, d) \
    *p++ = *(vos_uint8_t*)&d;\
	*p++ = *(((vos_uint8_t*)&d) + 1)

#define SET32BITS(p, d) \
    *p++ = *(vos_uint8_t*)&d ;\
	*p++ = *(((vos_uint8_t*)&d) + 1) ;\
	*p++ = *(((vos_uint8_t*)&d) + 2) ;\
	*p++ = *(((vos_uint8_t*)&d) + 3) 

EXT void MemeryCopy(void *src, void *dst, int n);
EXT int StringCopy(void *src, void *dst);


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

VOS_END_DECL


#endif			//__VOS_BS_TYPE_H__
