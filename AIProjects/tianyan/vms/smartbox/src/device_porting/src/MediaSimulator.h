#ifndef __MEDIA_SIMULATOR_H__
#define __MEDIA_SIMULATOR_H__


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



typedef struct
{
	AVFormatContext *pIfmt_ctx;
	AVStream *pVideoStream;
	AVStream *pAudioStream;
	int      iVideoIndex;
	int      iAudioIndex;
	unsigned char   aVideoHeader[256];
	int             iVideoHeaderSize;
	int             iVideoWidth;
	int             iVideoHeight;
	AVCodecID       eVideoCodecID;
	int      iH264NalSizeLen;

	AVCodecID       eAudioCodecID;
	int             iAudioSampleRate;
	int             iAudioChannels;
	unsigned char   aAudioHeader[256];
	unsigned char   iAudioHeaderSize;
} S_Media_Simulator;


void InitMediaSimulator(S_Media_Simulator*  pMedaiSimu);
void UnInitMediaSimulator(S_Media_Simulator*  pMedaiSimu);
int  OpenMedia(S_Media_Simulator*  pMediaSimu, char* pMediaURL);
int  ReadFrame(S_Media_Simulator*  pMediaSimu, void*  pOutput);
int  CloseMedia(S_Media_Simulator*  pMediaSimu);
int  DumpFrame(void*  pOutput);
#endif