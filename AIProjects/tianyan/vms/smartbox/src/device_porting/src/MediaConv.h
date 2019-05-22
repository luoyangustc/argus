#ifndef __MEDIA_CONV_H__
#define __MEDIA_CONV_H__

#ifdef _WIN32

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

extern "C"
{
#define  __STDC_CONSTANT_MACROS
#define _snprintf snprintf

#include "libavutil/timestamp.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/fifo.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"
};
#else  
//Linux...
#ifdef __cplusplus  
extern "C"
{
#endif  
#include "libavutil/timestamp.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/mem.h"
#include "libavutil/fifo.h"
#include "libavutil/error.h"
#include "libswscale/swscale.h"
#ifdef __cplusplus  
};
#endif  
#endif  


#define ERR_INVALID_PARAMETERS    0x80000001
#define ERR_CANNOT_FIND_VIDEO              0x80000002
#define ERR_CANNOT_FIND_VIDEO_DECODER              0x80000003
#define ERR_CANNOT_OPEN_VIDEO_DECODER              0x80000004
#define ERR_CANNOT_OPEN_VIDEO_SCALE                0x80000005
#define ERR_CANNOT_OPEN_MJPEG                      0x80000006
#define ERR_CANNOT_OPEN_MJPEG_ENCODER              0x80000007
#define ERR_CANNOT_ALLOC_MEMORY                    0x80000008
#define ERR_ENCODING_JPG                           0x80000009
#define ERR_OPEN_MEDIA_FAIL                        0x8000000a
#define ERR_OPEN_VIDEO_DECODER_FAIL                0x8000000b
#define ERR_SEEK_VIDEO_ERROR                       0x8000000c
#define ERR_READ_LIVE_VIDEO_ERROR                  0x8000000d
#define ERR_CREATE_VIDEO_STREAM_ERROR              0x8000000e
#define ERR_CREATE_AUDIO_STREAM_ERROR              0x8000000f


typedef struct
{
	char  strInputURL[1024];
	AVFormatContext *pIfmtCtx;
	int    iInputVideoIdx;
	int    iInputAudioIdx;

	char strOutputURL[1024];
	char strOutputFmt[128];
	AVFormatContext *pOfmtCtx;
	int iOutputVideoIdx;
	int iOutputAudioIdx;
	void*   pOutputPrivate;
	int     iConvertVideoForH264;
	int     iNalLengthSize;
	int     iConvertAudioForAAC;
	int     iRunFlag;
	int     iIndexForPrivate;
	int     iIndexForPrivateSub;
}S_Media_Conv;


int InitMediaConv(S_Media_Conv*   pMediaConv, char*  pInputURL, char* pOutputURL, char* pOutputFmt, void*  pPrivate, int iIndexForPrivate, int iIndexForPrivateSub);
int DoMediaConv(S_Media_Conv* pMediaConv);
int UnInitMediaConv(S_Media_Conv*   pMediaConv);
int PeekStreamURL(char*  pInputURL, void*  pMediaInfoPriv);
void* ThreadFuncForMediaConv(void*   pMediaConv);

#endif