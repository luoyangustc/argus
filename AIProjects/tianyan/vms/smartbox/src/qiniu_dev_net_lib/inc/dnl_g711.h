#ifndef __DNL_G711_H__
#define __DNL_G711_H__

int g711a_decode(void *pout_buf, int *pout_len, const void *pin_buf, const int in_len);
int g711a_encode(void *pout_buf, int *pout_len, const void *pin_buf, const int in_len);

#endif