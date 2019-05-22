#ifndef __ULU_AUDIO_H__
#define __ULU_AUDIO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "../protocol/protocol_device.h"

int ulu_audio_init();
int ulu_audio_open();
int ulu_audio_talk_msg_queue_add(Audio_Info_struct *msg);
int ulu_audio_talk_msg_queue_get(Audio_Info_struct *msg);

int ulu_audio_talk_data_queue_add(unsigned char *data, int len);
int ulu_audio_talk_data_queue_get(unsigned char *data, int *len);
int ulu_audio_talk_data_queue_free();
int ulu_audio_talk_data_queue_buf_size();
int ulu_audio_talk_data_recv_offset_get();
void ulu_audio_talk_data_recv_offset_set(int offset);
void ulu_audio_dell();

#ifdef __cplusplus
}
#endif

#endif	//__ULU_AUDIO_H__



