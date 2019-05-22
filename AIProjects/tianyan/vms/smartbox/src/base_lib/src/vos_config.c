#define __CONFIG_C__

#include "vos_config.h"
#include "vos_log.h"

#define VOS_MAKE_VERSION3_1(a,b,d) 	#a "." #b d
#define VOS_MAKE_VERSION3_2(a,b,d)	VOS_MAKE_VERSION3_1(a,b,d)

#define VOS_MAKE_VERSION4_1(a,b,c,d) 	#a "." #b "." #c d
#define VOS_MAKE_VERSION4_2(a,b,c,d)	VOS_MAKE_VERSION4_1(a,b,c,d)

#if VOS_VERSION_NUM_REV
const char* VOS_VERSION = VOS_MAKE_VERSION4_2(VOS_VERSION_NUM_MAJOR,
						         VOS_VERSION_NUM_MINOR,
						         VOS_VERSION_NUM_REV,
						         VOS_VERSION_NUM_EXTRA);
#else
const char* VOS_VERSION = VOS_MAKE_VERSION3_2(VOS_VERSION_NUM_MAJOR,
						         VOS_VERSION_NUM_MINOR,
						         VOS_VERSION_NUM_EXTRA);
#endif


const char* vos_get_version(void)
{
    return VOS_VERSION;
}

void vos_dump_config(void)
{

}


