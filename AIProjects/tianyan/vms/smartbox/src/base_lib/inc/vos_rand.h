
#ifndef __RAND_H__
#define __RAND_H__

#include "vos_config.h"
#include <stdlib.h>

#undef	EXT
#ifdef __RAND_C__  
#define EXT
#else
#define EXT extern
#endif

VOS_BEGIN_DECL

#define platform_srand    srand

#if defined(RAND_MAX) && RAND_MAX <= 0xFFFF
int platform_rand(void)
{
    return ((rand() & 0xFFFF) << 16) | (rand() & 0xFFFF);
}
#else
#define platform_rand   rand
#endif//RAND_MAX

EXT void vos_srand(unsigned int seed);
EXT int vos_rand(void);

VOS_END_DECL

#endif	/* __VOS_RAND_H__ */


