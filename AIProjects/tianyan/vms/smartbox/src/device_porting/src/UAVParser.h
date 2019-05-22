/*******************************************************************************
	File:		UAVParser.h

	Contains:	The audio and video parser header file.

	Written by:	Bangfei Jin

	Change History (most recent first):
	2016-12-08		Bangfei			Create file

*******************************************************************************/
#ifndef __UAVParser_H__
#define __UAVParser_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "CBitReader.h"

enum {
    kAVCProfileBaseline      = 0x42,
    kAVCProfileMain          = 0x4d,
    kAVCProfileExtended      = 0x58,
    kAVCProfileHigh          = 0x64,
    kAVCProfileHigh10        = 0x6e,
    kAVCProfileHigh422       = 0x7a,
    kAVCProfileHigh444       = 0xf4,
    kAVCProfileCAVLC444Intra = 0x2c
};

#define XRAW_IS_ANNEXB(p) ( !(*((p)+0)) && !(*((p)+1)) && (*((p)+2)==1))
#define XRAW_IS_ANNEXB2(p) ( !(*((p)+0)) && !(*((p)+1)) && !(*((p)+2))&& (*((p)+3)==1))

unsigned int qcAV_ParseUE (CBitReader *br);

// Optionally returns sample aspect ratio as well.
void qcAV_FindAVCDimensions(unsigned char * pBuffer, int nSize, int *width, int *height, int* numRef,int *sarWidth = NULL, int *sarHeight = NULL);

void qcAV_FindHEVCDimensions( unsigned char* buffer, unsigned int size, int *width, int *height);

bool qcAV_IsAVCReferenceFrame(unsigned char * pBuffer, int nSize);

bool qcAV_IsHEVCReferenceFrame(unsigned char* pBuffer, int nSize);

int qcAV_ConvertAVCNalHead (unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength);

int qcAV_ConvertHEVCNalHead (unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength);

int qcAV_ConvertAVCNalFrame (unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int nNalLength, int &IsKeyFrame, int nType = 0);

int qcAV_ParseAACConfig ( unsigned char* pBuffer, unsigned int size, int *out_sampling_rate, int *out_channels);

int qcAV_ConstructAACHeader(unsigned char* pBuffer, unsigned int size, int in_sampling_rate, int in_channels, int in_framesize);

int qcAV_GetAACFrameSize (unsigned char* pBuffer, unsigned int size, int *frame_size, int *out_sampling_rate, int *out_channels);

int qcAV_GetMPEGAudioFrameSize (unsigned char* pBuf, unsigned int *frame_size, int *out_sampling_rate, int *out_channels, int *out_bitrate, int *out_num_samples);

int qcAV_FindH264SpsPps(unsigned char * pBuffer, int nSize, unsigned char*  pBufSps, int iBufSpsMax, int& iSpsSize, unsigned char*  pBufPps, int iBufPpsMax, int& iPpsSize);

int qcAV_FindHEVCVpsSpsPps(unsigned char * pBuffer, int nSize, unsigned char*  pBufVps, int iBufVpsMax, int& iVpsSize, unsigned char*  pBufSps, int iBufSpsMax, int& iSpsSize, unsigned char*  pBufPps, int iBufPpsMax, int& iPpsSize);

int qcAV_ParseADTSAACHeaderInfo(unsigned char * pBuffer, int nSize, int *pOut_sampling_rate, int *pOut_channels, int *pOut_samplebitCount);

int qcAV_IsNalUnit(unsigned char* pBuffer, int nSize);

int qcAV_ConvertHEVCNalHead2 (unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength);

#endif  // __UAVParser_H__
