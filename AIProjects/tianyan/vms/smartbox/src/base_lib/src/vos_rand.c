#define __RAND_C__

#include "vos_rand.h"

void vos_srand(unsigned int seed)
{
    platform_srand(seed);
}

int vos_rand(void)
{
    return platform_rand();
}


