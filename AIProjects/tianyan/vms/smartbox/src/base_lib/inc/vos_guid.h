#ifndef __VOS_GUID_H__
#define __VOS_GUID_H__

#include "vos_types.h"

#if (OS_WIN32 == 1)
#pragma comment(lib,"ole32.lib")
#endif

VOS_BEGIN_DECL

extern const unsigned VOS_GUID_STRING_LENGTH;

#define VOS_GUID_MAX_LENGTH  36

unsigned vos_GUID_STRING_LENGTH(void);
vos_str_t* vos_generate_unique_string(vos_str_t *str);
void vos_create_unique_string(vos_str_t *str);

VOS_END_DECL

#endif/* __VOS_GUID_H__ */


