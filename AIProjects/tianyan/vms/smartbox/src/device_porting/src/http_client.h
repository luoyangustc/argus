#ifndef __HTTP_CLIENT_H__
#define __HTTP_CLIENT_H__

int HttpGet(char*  pHttpGetURL, unsigned char**  ppOutput, int* piOutputMaxSize, int*  piOutputSize);

#endif