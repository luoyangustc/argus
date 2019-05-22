
#ifndef __DNL_DEV_H__
#define __DNL_DEV_H__

#include "comm_includes.h"
#include "dnl_def.h"
#include "dnl_util.h"
#include "DeviceSDK.h"

#undef  EXT
#ifndef __DNL_DEV_C__
#define EXT extern
#else
#define EXT
#endif

#define LOCAL_PORT_VLU  5000

#define UDP_TYPE    0
#define TCP_TYPE    1

VOS_BEGIN_DECL

EXT Dev_Info_t g_DnlDevInfo; /* 设备的全局信息 */
EXT vos_mutex_t* g_DnlDevInfoMutex;

EXT void dnl_dev_init(void);

VOS_END_DECL

#endif


