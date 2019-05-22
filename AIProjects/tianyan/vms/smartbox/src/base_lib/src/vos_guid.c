#include "vos_guid.h"
#include "vos_assert.h"
#include "vos_rand.h"
#include "vos_os.h"
#include "vos_string.h"

void vos_create_unique_string(vos_str_t *str)
{
    str->ptr = (char*)malloc(VOS_GUID_STRING_LENGTH);
    vos_generate_unique_string(str);
}

const unsigned VOS_GUID_STRING_LENGTH=32;

static char guid_chars[64];

unsigned vos_GUID_STRING_LENGTH()
{
    return VOS_GUID_STRING_LENGTH;
}

static void init_guid_chars(void)
{
    char *p = guid_chars;
    unsigned i;

    for (i=0; i<10; ++i)
        *p++ = '0'+i;

    for (i=0; i<26; ++i) {
        *p++ = 'a'+i;
        *p++ = 'A'+i;
    }

    *p++ = '-';
    *p++ = '.';
}

vos_str_t* vos_generate_unique_string(vos_str_t *str)
{
    char *p, *end;

    if (guid_chars[0] == '\0') {
        vos_enter_critical_section();
        if (guid_chars[0] == '\0') {
            init_guid_chars();
        }
        vos_leave_critical_section();
    }

    vos_assert(VOS_GUID_STRING_LENGTH % 2 == 0);

    for (p=str->ptr, end=p+VOS_GUID_STRING_LENGTH; p<end; ) {
        vos_uint32_t rand_val = vos_rand();
        vos_uint32_t rand_idx = RAND_MAX;

        for ( ; rand_idx>0 && p<end; rand_idx>>=8, rand_val>>=8, p++) {
            *p = guid_chars[(rand_val & 0xFF) & 63];
        }
    }

    str->slen = VOS_GUID_STRING_LENGTH;
    return str;
}




