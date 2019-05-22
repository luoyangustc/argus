/*******************************************************************************
	File:		UAVParser.cpp

	Contains:	The audio and video parser implement file.

	Written by:	Bangfei Jin

	Change History (most recent first):
	2016-12-08		Bangfei			Create file

*******************************************************************************/
#include "UAVParser.h"


unsigned int qcAV_ParseUE (CBitReader *br) 
{
    unsigned int numZeroes = 0;
    while (br->GetBits(1) == 0 && br->numBitsLeft() > 0) 
        ++numZeroes;

    unsigned int x = br->GetBits(numZeroes);

    return x + (1u << numZeroes) - 1;
}

int qcAV_ParseSE (CBitReader *br) 
{
    int codeNum = qcAV_ParseUE(br);
    return (codeNum & 1) ? (codeNum + 1) / 2 : -(codeNum / 2);
}

static void qcAV_SkipScalingList (CBitReader *br, unsigned int sizeOfScalingList) 
{
    unsigned int lastScale = 8;
    unsigned int nextScale = 8;
    for (unsigned int j = 0; j < sizeOfScalingList; ++j) 
	{
        if (nextScale != 0) 
		{
            signed int delta_scale = qcAV_ParseSE(br);
            nextScale = (lastScale + delta_scale + 256) % 256;
        }

        lastScale = (nextScale == 0) ? lastScale : nextScale;
    }
}

// Determine video dimensions from the sequence parameterset.
void qcAV_FindAVCDimensions(unsigned char * pBuffer, int nSize,
						int *width, int *height, int* numRef,
						int *sarWidth, int *sarHeight) 
{
	if (pBuffer[2]==0 && pBuffer[3]==1) {
		pBuffer+=5;
		nSize -= 5;
	} else {
		pBuffer+=4;
		nSize -= 4;
	}

    CBitReader br(pBuffer, nSize);
    unsigned int profile_idc = br.GetBits(8);
    br.SkipBits(16);
    qcAV_ParseUE(&br);  // seq_parameter_set_id

    unsigned int chroma_format_idc = 1;  // 4:2:0 chroma format

    if (profile_idc == 100 || profile_idc == 110
            || profile_idc == 122 || profile_idc == 244
            || profile_idc == 44 || profile_idc == 83 || profile_idc == 86) {
        chroma_format_idc = qcAV_ParseUE(&br);
        if (chroma_format_idc == 3) 
		{
            br.SkipBits(1);  // residual_colour_transform_flag
        }
        qcAV_ParseUE(&br);  // bit_depth_luma_minus8
        qcAV_ParseUE(&br);  // bit_depth_chroma_minus8
        br.SkipBits(1);  // qpprime_y_zero_transform_bypass_flag

		// seq_scaling_matrix_present_flag
        if (br.GetBits(1)) 
		{  
            for (size_t i = 0; i < 8; ++i) 
			{
				// seq_scaling_list_present_flag[i]
                if (br.GetBits(1)) 
				{ 
                    // WARNING: the code below has not ever been exercised...
                    // need a real-world example.
                    if (i < 6) {
                        // ScalingList4x4[i],16,...
                        qcAV_SkipScalingList(&br, 16);
                    } else {
                        // ScalingList8x8[i-6],64,...
                        qcAV_SkipScalingList(&br, 64);
                    }
                }
            }
        }
    }

    qcAV_ParseUE(&br);  // log2_max_frame_num_minus4
    unsigned int pic_order_cnt_type = qcAV_ParseUE(&br);

    if (pic_order_cnt_type == 0) 
	{
        qcAV_ParseUE(&br);  // log2_max_pic_order_cnt_lsb_minus4
    } else if (pic_order_cnt_type == 1) {
        // offset_for_non_ref_pic, offset_for_top_to_bottom_field and
        // offset_for_ref_frame are technically se(v), but since we are
        // just skipping over them the midpoint does not matter.

        br.GetBits(1);  // delta_pic_order_always_zero_flag
        qcAV_ParseUE(&br);  // offset_for_non_ref_pic
        qcAV_ParseUE(&br);  // offset_for_top_to_bottom_field

        unsigned int num_ref_frames_in_pic_order_cnt_cycle = qcAV_ParseUE(&br);
        for (unsigned int i = 0; i < num_ref_frames_in_pic_order_cnt_cycle; ++i) {
            qcAV_ParseUE(&br);  // offset_for_ref_frame
        }
    }

    unsigned int nRef = qcAV_ParseUE(&br);  // num_ref_frames
	if(numRef)
		*numRef = nRef;

    br.GetBits(1);  // gaps_in_frame_num_value_allowed_flag

    unsigned int pic_width_in_mbs_minus1 = qcAV_ParseUE(&br);
    unsigned int pic_height_in_map_units_minus1 = qcAV_ParseUE(&br);
    unsigned int frame_mbs_only_flag = br.GetBits(1);

    *width = pic_width_in_mbs_minus1 * 16 + 16;
    *height = (2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 * 16 + 16);

    if (!frame_mbs_only_flag) 
        br.GetBits(1);  // mb_adaptive_frame_field_flag

    br.GetBits(1);  // direct_8x8_inference_flag

	// frame_cropping_flag
    if (br.GetBits(1)) 
	{  
        unsigned int frame_crop_left_offset = qcAV_ParseUE(&br);
        unsigned int frame_crop_right_offset = qcAV_ParseUE(&br);
        unsigned int frame_crop_top_offset = qcAV_ParseUE(&br);
        unsigned int frame_crop_bottom_offset = qcAV_ParseUE(&br);

        unsigned int cropUnitX, cropUnitY;
        if (chroma_format_idc == 0  /* monochrome */) {
            cropUnitX = 1;
            cropUnitY = 2 - frame_mbs_only_flag;
        } else {
            unsigned int subWidthC = (chroma_format_idc == 3) ? 1 : 2;
            unsigned int subHeightC = (chroma_format_idc == 1) ? 2 : 1;

            cropUnitX = subWidthC;
            cropUnitY = subHeightC * (2 - frame_mbs_only_flag);
        }

        *width -= (frame_crop_left_offset + frame_crop_right_offset) * cropUnitX;
        *height -= (frame_crop_top_offset + frame_crop_bottom_offset) * cropUnitY;
    }

    if (sarWidth != NULL) 
        *sarWidth = 0;

    if (sarHeight != NULL) 
        *sarHeight = 0;

	// vui_parameters_present_flag
    if (br.GetBits(1)) 
	{  
        unsigned int sar_width = 0, sar_height = 0;

		// aspect_ratio_info_present_flag
        if (br.GetBits(1)) 
		{  
            unsigned int aspect_ratio_idc = br.GetBits(8);

			// extendedSAR 
            if (aspect_ratio_idc == 255 ) 
			{
                sar_width = br.GetBits(16);
                sar_height = br.GetBits(16);
            }
			else if (aspect_ratio_idc > 0 && aspect_ratio_idc < 14) 
			{
                static const int kFixedSARWidth[] = {
                    1, 12, 10, 16, 40, 24, 20, 32, 80, 18, 15, 64, 160
                };

                static const int kFixedSARHeight[] = {
                    1, 11, 11, 11, 33, 11, 11, 11, 33, 11, 11, 33, 99
                };

                sar_width = kFixedSARWidth[aspect_ratio_idc - 1];
                sar_height = kFixedSARHeight[aspect_ratio_idc - 1];
            }
        }

		if (sarWidth != NULL) {
            *sarWidth = sar_width;
        }

        if (sarHeight != NULL) {
            *sarHeight = sar_height;
        }
    }
}

int qcAV_AdjustSPS(unsigned char *sps, unsigned int*spsLen) 
{
    unsigned char *data = sps;
    unsigned int  size = *spsLen;
    unsigned int  offset = 0;

    while (offset + 2 <= size) 
	{
        if (data[offset] == 0x00 && data[offset+1] == 0x00 && data[offset+2] == 0x03) {
            //found 00 00 03
            if (offset + 2 == size) {//00 00 03 as suffix
                *spsLen -=1;
                return 0;
            }

            offset += 2; //point to 0x03
            memcpy(data+offset, data+(offset+1), size - offset);//cover ox03

            size -= 1;
            *spsLen -= 1;
            continue;
        }
        ++offset;
    }
    return 0;
}

void qcAV_HEVCParsePtl(CBitReader &br, unsigned int max_sub_layers_minus1)
{
    unsigned int i;
    unsigned char sub_layer_profile_present_flag[8];
    unsigned char sub_layer_level_present_flag[8];

    br.SkipBits(2);
    br.SkipBits(1);
    br.SkipBits(5);
    br.SkipBits(32);
    br.SkipBits(48);
    br.SkipBits(8);

    for (i = 0; i < max_sub_layers_minus1; i++) {
        sub_layer_profile_present_flag[i] = br.GetBits(1);
        sub_layer_level_present_flag[i]   = br.GetBits(1);
    }

	if (max_sub_layers_minus1 > 0) {
        for (i = max_sub_layers_minus1; i < 8; i++)
            br.GetBits(2); // reserved_zero_2bits[i]
	}

    for (i = 0; i < max_sub_layers_minus1; i++) {
        if (sub_layer_profile_present_flag[i]) {
           
           //sub_layer_profile_space[i]                     u(2)
           //sub_layer_tier_flag[i]                         u(1)
           //sub_layer_profile_idc[i]                       u(5)
           //sub_layer_profile_compatibility_flag[i][0..31] u(32)
           //sub_layer_progressive_source_flag[i]           u(1)
           //sub_layer_interlaced_source_flag[i]            u(1)
           //sub_layer_non_packed_constraint_flag[i]        u(1)
           //sub_layer_frame_only_constraint_flag[i]        u(1)
           //sub_layer_reserved_zero_44bits[i]              u(44)
           
            br.SkipBits(32);
            br.SkipBits(32);
            br.SkipBits(24);
        }

        if (sub_layer_level_present_flag[i])
            br.SkipBits(8);
    }
}

void qcAV_FindHEVCDimensions (unsigned char* buffer, unsigned int size, int *width, int *height)
{
	unsigned int sps_max_sub_layers_minus1;
	if (buffer[2]==0 && buffer[3]==1) 
	{
		buffer+=6;
		size -= 6;
	} 
	else if(buffer[1]==0 && buffer[2]==1)
	{
		buffer+=5;
		size -= 5;
	} else {
		buffer+=2;
		size -= 2;
	}

	qcAV_AdjustSPS (buffer, &size);

    CBitReader br(buffer, size);

    br.SkipBits(4); // sps_video_parameter_set_id
    sps_max_sub_layers_minus1 = br.GetBits(3);

    br.GetBits(1); //sps_temporal_id_nesting_flag
    qcAV_HEVCParsePtl(br, sps_max_sub_layers_minus1);

    qcAV_ParseUE(&br); // sps_seq_parameter_set_id

    int chroma_format_idc = qcAV_ParseUE(&br);
	int separate_colour_plane_flag = 0;

	if (chroma_format_idc == 3) {
        separate_colour_plane_flag = br.GetBits(1); 
	}

    int pic_width_in_luma_samples = qcAV_ParseUE(&br); // pic_width_in_luma_samples
    int pic_height_in_luma_samples = qcAV_ParseUE(&br); // pic_height_in_luma_samples

    int conformance_window_flag = br.GetBits(1);
	int conf_win_left_offset = 0;
    int conf_win_right_offset = 0;
    int conf_win_top_offset = 0;
    int conf_win_bottom_offset = 0;
	if (conformance_window_flag) {        // conformance_window_flag
        conf_win_left_offset = qcAV_ParseUE(&br); // conf_win_left_offset
        conf_win_right_offset = qcAV_ParseUE(&br); // conf_win_right_offset
        conf_win_top_offset = qcAV_ParseUE(&br); // conf_win_top_offset
        conf_win_bottom_offset = qcAV_ParseUE(&br); // conf_win_bottom_offset
    }

    //int bitDepthLumaMinus8          = qcAV_ParseUE(&br);
    //int bitDepthChromaMinus8        = qcAV_ParseUE(&br);
    //log2_max_pic_order_cnt_lsb_minus4 = qcAV_ParseUE(&br);

    // sps_sub_layer_ordering_info_present_flag 
    //i = br.GetBits(1) ? 0 : sps_max_sub_layers_minus1;
	//for (; i <= sps_max_sub_layers_minus1; i++) {
    //    qcAV_ParseUE(&br); // max_dec_pic_buffering_minus1
	//	qcAV_ParseUE(&br); // max_num_reorder_pics
	//	qcAV_ParseUE(&br); // max_latency_increase_plus1
	//}

    //qcAV_ParseUE(&br); // log2_min_luma_coding_block_size_minus3
    //qcAV_ParseUE(&br); // log2_diff_max_min_luma_coding_block_size
    //qcAV_ParseUE(&br); // log2_min_transform_block_size_minus2
    //qcAV_ParseUE(&br); // log2_diff_max_min_transform_block_size
    //qcAV_ParseUE(&br); // max_transform_hierarchy_depth_inter
    //qcAV_ParseUE(&br); // max_transform_hierarchy_depth_intra

	int sub_width_c  = ((1==chroma_format_idc)||(2 == chroma_format_idc))&&(0==separate_colour_plane_flag)?2:1;
	int sub_height_c = (1==chroma_format_idc)&& (0 == separate_colour_plane_flag)?2:1;
	int nWidth  = pic_width_in_luma_samples;
	int nHeight = pic_height_in_luma_samples;

	nWidth  -= (sub_width_c*conf_win_right_offset + sub_width_c*conf_win_left_offset);
	nHeight -= (sub_height_c*conf_win_bottom_offset + sub_height_c*conf_win_top_offset);

	if(width) 
	{
		*width = nWidth;
		*height = nHeight;
	}
}

bool qcAV_IsAVCReferenceFrame(unsigned char * pBuffer, int nSize)
{
	if (pBuffer[2]==0 && pBuffer[3]==1) {
		pBuffer+=4;
		nSize -= 4;
	} else {
		pBuffer+=3;
		nSize -= 3;
	}

	int naluType = pBuffer[0]&0x0f;
	int isRef	 = 1;
	while(naluType!=1 && naluType!=5)//find next NALU
	{
		unsigned char * p = pBuffer;  
		unsigned char * endPos = pBuffer+nSize;
		for (; p < endPos; p++) {
			if (XRAW_IS_ANNEXB(p))	{
				nSize  -= p-pBuffer;
				pBuffer = p+3;
				naluType = pBuffer[0]&0x0f;
				break;
			}

			if (XRAW_IS_ANNEXB2(p))	{
				nSize  -= p-pBuffer;
				pBuffer = p+4;
				naluType = pBuffer[0]&0x0f;
				break;
			}
		}

		if(p>=endPos)
			return false; 
	}
	
	if(naluType == 5)
		return true;

	if(naluType==1)	{
		isRef = (pBuffer[0]>>5) & 3;
	}

	return (isRef != 0);
}

int qcAV_ConvertAVCNalHead(unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength)
{
	if (pOutBuffer == NULL || pInBuffer == NULL)
		return -1;
	
	if (nInSize < 12)
		return -1;

	//char configurationVersion = pInBuffer[0];
	//char AVCProfileIndication = pInBuffer[1];
	//char profile_compatibility = pInBuffer[2];
	//char AVCLevelIndication  = pInBuffer[3];

	nNalLength =  (pInBuffer[4]&0x03)+1;
	int nNalWord = 0x01000000;
	if (nNalLength == 3)
		nNalWord = 0X010000;

	int nNalLen = nNalLength;
	if (nNalLength < 3)	{
		nNalLen = 4;
	}

	int HeadSize = 0;
	int i = 0;

	int nSPSNum = pInBuffer[5]&0x1f;
	unsigned char* pBuffer = pInBuffer + 6;

	for (i = 0; i< nSPSNum; i++)
	{
		int nSPSLength = (pBuffer[0]<<8)| pBuffer[1];
		pBuffer += 2;

		memcpy (pOutBuffer + HeadSize, &nNalWord, nNalLen);
		HeadSize += nNalLen;

		if(nSPSLength > (nInSize - (pBuffer - pInBuffer))){
			return -1;
		}

		memcpy (pOutBuffer + HeadSize, pBuffer, nSPSLength);
		HeadSize += nSPSLength;
		pBuffer += nSPSLength;
	}

	int nPPSNum = *pBuffer++;
	for (i=0; i< nPPSNum; i++)
	{
		int nPPSLength = (pBuffer[0]<<8) | pBuffer[1];
		pBuffer += 2;
		
		memcpy (pOutBuffer + HeadSize, &nNalWord, nNalLen);
		HeadSize += nNalLen;
		
		if(nPPSLength > (nInSize - (pBuffer - pInBuffer))){
			return -1;
		}

		memcpy (pOutBuffer + HeadSize, pBuffer, nPPSLength);
		HeadSize += nPPSLength;
		pBuffer += nPPSLength;
	}

	nOutSize = HeadSize;

	return 0;
}

int qcAV_ConvertHEVCNalHead(unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength)
{
	if (pOutBuffer == NULL || pInBuffer == NULL)
		return -1;

	if (nInSize < 22)
		return -1;
    
	unsigned char * pData = pInBuffer;
	nNalLength =  (pData[21]&0x03)+1;
	int nNalLen = nNalLength;
	if (nNalLength < 3)	{
		nNalLen = 4;
	}

	unsigned int nNalWord = 0x01000000;
	if (nNalLength == 3)
		nNalWord = 0X010000;

	int nHeadSize = 0;
	unsigned char * pBuffer = pOutBuffer;
	int nArrays = pData[22];
	int nNum = 0;

	pData += 23;
	if(nArrays)
	{
		for(nNum = 0; nNum < nArrays; nNum++)
		{
			unsigned char nal_type = 0;
			nal_type = pData[0]&0x3F;
			pData += 1;
			switch(nal_type)
			{
			case 33://sps
				{
					int nSPSNum = (pData[0] << 8)|pData[1];
					pData += 2;
					for(int i = 0; i < nSPSNum; i++)
					{
						memcpy (pBuffer + nHeadSize, &nNalWord, nNalLen);
						nHeadSize += nNalLen;
						int nSPSLength = (pData[0] << 8)|pData[1];
						pData += 2;
						if(nSPSLength > (nInSize - (pData - pInBuffer))){
							nOutSize = 0;
							return -1;
						}

						memcpy (pBuffer + nHeadSize, pData, nSPSLength);
						nHeadSize += nSPSLength;
						pData += nSPSLength;
					}
				}
				break;
			case 34://pps
				{
					int nPPSNum = (pData[0] << 8) | pData[1];
					pData += 2;
					for(int i = 0; i < nPPSNum; i++)
					{
						memcpy (pBuffer + nHeadSize, &nNalWord, nNalLen);
						nHeadSize += nNalLen;
						int nPPSLength = (pData[0] << 8)| pData[1];
						pData += 2;
						if(nPPSLength > (nInSize - (pData - pInBuffer))){
							nOutSize = 0;
							return -1;
						}
						memcpy (pBuffer + nHeadSize, pData, nPPSLength);
						nHeadSize += nPPSLength;
						pData += nPPSLength;
					}
				}
				break;
			case 32: //vps
				{
					int nVPSNum = (pData[0] << 8 )| pData[1] ;
					pData += 2;
					for(int i = 0; i < nVPSNum; i++)
					{
						memcpy (pBuffer + nHeadSize, &nNalWord, nNalLen);
						nHeadSize += nNalLen;
						int nVPSLength = (pData[0] << 8 )|pData[1];
						pData += 2;
						if(nVPSLength > (nInSize - (pData - pInBuffer))){
							nOutSize = 0;
							return -1;
						}
						memcpy (pBuffer + nHeadSize, pData, nVPSLength);
						nHeadSize += nVPSLength;
						pData += nVPSLength;
					}
				}
				break;
			default://just skip the data block
				{
					int nSKP = (pData[0] << 8 )|pData[1];
					pData += 2;
					for(int i = 0; i < nSKP; i++)
					{
						int nAKPLength = (pData[0] << 8) | pData[1];
						if(nAKPLength > (nInSize - (pData - pInBuffer))){
							nOutSize = 0;
							return -1;
						}
						pData += 2;
						pData += nAKPLength;
					}

				}
				break;
			}
		}
	}

	nOutSize = nHeadSize;

	return 0;
}

int qcAV_ConvertAVCNalFrame(unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int nNalLength, int &IsKeyFrame, int nType)
{
	unsigned char*  pBuffer = pInBuffer;
	int	nNalLen = 0;
	int	nNalType = 0;	
	int nNalWord = 0x01000000;
	if (nNalLength == 3)
		nNalWord = 0X010000;

	if(nNalLength == 0) {
		return -1;
	}

	int i = 0;
	int leftSize = nInSize;
	nOutSize = 0;

	while (pBuffer - pInBuffer + 4 < nInSize)
	{
		nNalLen = *pBuffer++;
		for (i = 0; i < (int)nNalLength - 1; i++)
		{
			nNalLen = nNalLen << 8;
			nNalLen += *pBuffer++;
		}

		if(nNalType != 1 && nNalType != 5) {
			if(nType == 12) {
				nNalType = (pBuffer[0] >> 1)&0x3f;
			} else {
				nNalType = pBuffer[0]&0x0f;
			}
		}

		leftSize -= nNalLength;

		if(nNalLen > leftSize || nNalLen <= 0)
		{
			nOutSize = 0;
			return -1;
		}

		if (nNalLength == 3 || nNalLength == 4)
		{
			memcpy ((pBuffer - nNalLength), &nNalWord, nNalLength);
		}
		else
		{
			memcpy (pOutBuffer + nOutSize, &nNalWord, 4);
			nOutSize += 4;
			memcpy (pOutBuffer + nOutSize, pBuffer, nNalLen);
			nOutSize += nNalLen;
		}

		leftSize -= nNalLen;
		pBuffer += nNalLen;
	}

	if(nType == 12) {
		if(nNalType >= 19 && nNalType <= 21)
			IsKeyFrame = 1;
	} else {
		if(nNalType == 5)
			IsKeyFrame = 1;
	}

	return 0;
}

static const int qcAV_AACSampleRate[] = {
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
        16000, 12000, 11025, 8000};

int qcAV_ParseAACConfig (unsigned char* pBuffer, unsigned int size, int *out_sampling_rate, int *out_channels)
{
	if(pBuffer == NULL || size < 2) {
		return -1;
	}

	int sampleIndex = -1;
	int sampFreq = 0;
	int channel = 0;

	//int object = pBuffer[0] >> 3;
	sampleIndex = ((pBuffer[0] & 7) << 1) | (pBuffer[1] >> 7);
	if(sampleIndex == 0x0f) {
		if(size < 5)
			return -1;

		sampFreq = ((pBuffer[1]&0x7f) << 17) | (pBuffer[2] << 9) | ((pBuffer[3] << 1)) | (pBuffer[4] >> 7);

		channel = (pBuffer[4]&0x78) >> 3;
	} else {
		channel = (pBuffer[1]&0x78) >> 3;
		sampFreq = qcAV_AACSampleRate[sampleIndex];
	}

	if(out_sampling_rate) {
		*out_sampling_rate = sampFreq;
	}

	if(out_channels) {
		*out_channels = channel;
	}

	return 0;
}

int qcAV_ConstructAACHeader (unsigned char* pBuffer, unsigned int size, int in_sampling_rate, int in_channels, int in_framesize)
{
	if(pBuffer == NULL || size < 7) {
		return -1;
	}

	int sampleIndex = -1;
	int i;
	for (i = 0; i < 12; i++) {
		if (in_sampling_rate == qcAV_AACSampleRate[i]) {
			sampleIndex = i;
			break;
		}	
	}

	if(sampleIndex == -1) {
		return -1;
	}
	
	pBuffer[0] = 0xFF;
	pBuffer[1] = 0xF9;

	pBuffer[2] = 0x40 | (unsigned char)(sampleIndex << 2) |((unsigned char)(in_channels >> 2) & 0x01);
	pBuffer[3] = (unsigned char)((in_channels << 6) & 0xc0) | (0x01 << 3) | (unsigned char)(((in_framesize + 7) >> 11) & 0x03); 
	pBuffer[4] = (unsigned char)((unsigned short)((in_framesize + 7) >> 3) & 0x00ff);
	pBuffer[5] = (unsigned char)((unsigned char)((in_framesize + 7) & 0x07) << 5) | 0x1f;
	pBuffer[6] = 0xF8;

	return 7;
}

int GetAACFrameSize (unsigned char* pBuffer, unsigned int size, int *frame_size, int *out_sampling_rate, int *out_channels)
{
	int framelen = 0;
	int sampleIndex, profile, channel;
	int maxFrameLen = 2048;
	int aSync = 1;
	unsigned char *pBuf = pBuffer;

	if (out_sampling_rate) {
        *out_sampling_rate = 0;
    }

    if (out_channels) {
        *out_channels = 0;
    }

	if(frame_size) {
		*frame_size = 0;
	}

	if(pBuf == NULL) {
		return -1;
	}

	do {
		int  i;
		int  inLen = size;

		for (i = 0; i < inLen - 1; i++) {			
			if ( (pBuf[0] & 0xFF) == 0xFF && (pBuf[1] & 0xF0) == 0xF0 )
				break;

			pBuf++;
			inLen--;
			if (inLen <= 7)	{
				return -1;
			}
		}

		framelen = ((pBuf[3] & 0x3) << 11) + (pBuf[4] << 3) + (pBuf[5] >> 5);
		sampleIndex = (pBuf[2] >> 2) &0xF;
		profile = (pBuf[2] >> 6) + 1;
		channel = ((pBuf[2]&0x01) << 2) | (pBuf[3] >> 6);

		if(framelen > maxFrameLen || profile > 2 || channel > 6 || sampleIndex > 12) {
			pBuf++;
			inLen--;
			continue;
		}

		if(framelen > inLen || inLen == 0){
			return -1;
		}

		if(framelen > inLen) {
			pBuf++;
			inLen--;
			continue;		
		}

		if(framelen + 2 < inLen) {
			if(pBuf[framelen] == 0xFF && (pBuf[framelen + 1] & 0xF0) == 0xF0) {
				aSync = 0;
			}				
		}

		if(framelen == inLen){
			aSync = 0;
		}
	}while(aSync);

	if(frame_size) {
		*frame_size = pBuf - pBuffer + framelen;
	}

	if (out_sampling_rate) {
        *out_sampling_rate = qcAV_AACSampleRate[sampleIndex];
    }

    if (out_channels) {
       *out_channels = channel;
    }

	return 0;
}

int qcAV_GetMPEGAudioFrameSize (unsigned char* pBuf, unsigned int *frame_size, int *out_sampling_rate, int *out_channels, int *out_bitrate, int *out_num_samples) 
{
	if(frame_size) {
		*frame_size = 0;
	}

    if (out_sampling_rate) {
        *out_sampling_rate = 0;
    }

    if (out_channels) {
        *out_channels = 0;
    }

    if (out_bitrate) {
        *out_bitrate = 0;
    }

    if (out_num_samples) {
        *out_num_samples = 1152;
    }

	if(pBuf == NULL) {
		return -1;
	}

	unsigned int verIdx		= (pBuf[1] >> 3) & 0x03;
	unsigned int version    = (pBuf[1] >> 3) & 0x03;
	unsigned int layer		=  4 - ((pBuf[1] >> 1) & 0x03);     
	unsigned int brIdx		= (pBuf[2] >> 4) & 0x0f;
	unsigned int srIdx		= (pBuf[2] >> 2) & 0x03;
	unsigned int paddingBit	= (pBuf[2] >> 1) & 0x01;
	unsigned int mode	    = (pBuf[3] >> 6) & 0x03;

	if (srIdx == 3 || brIdx == 15 || verIdx == 1)
		return -1;

    if (layer == 0x04) {
        layer -= 1;
    }

    static const int kSamplingRateV1[] = { 44100, 48000, 32000 };
    int sampling_rate = kSamplingRateV1[srIdx];
	// v2
    if (version == 2) {
        sampling_rate /= 2;
    }
	// V2.5
	else if (version == 0) 
	{
        sampling_rate /= 4;
    }

     if (layer == 1) {
        // layer I
        static const int kBitrateV1[] = {
            0, 32, 64, 96, 128, 160, 192, 224, 256,
            288, 320, 352, 384, 416, 448
        };

        static const int kBitrateV2[] = {
            0, 32, 48, 56, 64, 80, 96, 112, 128,
            144, 160, 176, 192, 224, 256
        };

		// V1
        int bitrate = (version == 3 ) ? kBitrateV1[brIdx] : kBitrateV2[brIdx];
        if (out_bitrate) {
            *out_bitrate = bitrate;
        }

        *frame_size = (12000 * bitrate / sampling_rate + paddingBit) * 4;

        if (out_num_samples) {
            *out_num_samples = 384;
        }
    } else {
        // layer II or III
        static const int kBitrateV1L2[] = {
            0, 32, 48, 56, 64, 80, 96, 112, 128,
            160, 192, 224, 256, 320, 384
        };

        static const int kBitrateV1L3[] = {
            0, 32, 40, 48, 56, 64, 80, 96, 112,
            128, 160, 192, 224, 256, 320
        };

        static const int kBitrateV2[] = {
            0, 8, 16, 24, 32, 40, 48, 56, 64,
            80, 96, 112, 128, 144, 160
        };

        int bitrate;
		// V1
        if (version == 3)
		{
			// L2
            bitrate = (layer == 2) ? kBitrateV1L2[brIdx] : kBitrateV1L3[brIdx];

            if (out_num_samples) {
                *out_num_samples = 1152;
            }
        } else {
            // V2 (or 2.5)
            bitrate = kBitrateV2[brIdx];
			// L3
            if (out_num_samples) {
                *out_num_samples = (layer == 3) ? 576 : 1152;
            }
        }

        if (out_bitrate) {
            *out_bitrate = bitrate;
        }

		// V1
        if (version == 3) {
            *frame_size = 144000 * bitrate / sampling_rate + paddingBit;
        } else {
            // V2 or V2.5 , L3
            unsigned int tmp = (layer == 1) ? 72000 : 144000;
            *frame_size = tmp * bitrate / sampling_rate + paddingBit;
        }
    }

    if (out_sampling_rate) {
        *out_sampling_rate = sampling_rate;
    }

    if (out_channels) {
        *out_channels = (mode == 3) ? 1 : 2;
    }

    return 0;
}

int qcAV_FindH264SpsPps(unsigned char * pBuffer, int nSize, unsigned char*  pBufSps, int iBufSpsMax, int& iSpsSize, unsigned char*  pBufPps, int iBufPpsMax, int& iPpsSize)
{
	unsigned char aStartCode[4] = {0, 0, 0, 1};
	int ioffset = 0;
	unsigned char * pScan = pBuffer;
	unsigned char * pScanEnd = pBuffer + nSize - 4;
	unsigned char * pFindNalDataStart = NULL;
	unsigned char * pFindNalDataEnd = NULL;
	unsigned char   naluType = 0;
	
	iSpsSize = 0;
	iPpsSize = 0;

	while (pScan < pScanEnd)
	{
		if (iSpsSize != 0 && iPpsSize != 0)
		{
			break;
		}

		if (memcmp(pScan, aStartCode + 1, 3) == 0)
		{
			if (pFindNalDataStart != NULL)
			{
				if ((pScan > pBuffer) && *(pScan - 1) == 0)
				{
					pFindNalDataEnd = pScan - 1;
				}
				else
				{
					pFindNalDataEnd = pScan;
				}
			}

			if (naluType == 7 &&  iSpsSize == 0)
			{
				if (iBufSpsMax > (4 + pFindNalDataEnd - pFindNalDataStart))
				{
					memcpy(pBufSps, aStartCode, 4);
					memcpy(pBufSps + 4, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
					iSpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
				}
				else
				{
					return 1;
				}
			}

			if (naluType == 8 && iPpsSize == 0)
			{
				if (iBufPpsMax > (4 + pFindNalDataEnd - pFindNalDataStart))
				{
					memcpy(pBufPps, aStartCode, 4);
					memcpy(pBufPps + 4, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
					iPpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
				}
				else
				{
					return 1;
				}
			}

			naluType = (*(pScan + 3)) & 0x0f;
			switch (naluType)
			{
				case 7:
				case 8:
				{
					pFindNalDataStart = pScan + 3;
					break;
				}

				default:
				{
					break;
				}
			}

			pScan += 3;
		}
		else
		{
			pScan++;
		}
	}

	return 0;
}


int qcAV_FindHEVCVpsSpsPps(unsigned char * pBuffer, int nSize, unsigned char*  pBufVps, int iBufVpsMax, int& iVpsSize,
	unsigned char*  pBufSps, int iBufSpsMax, int& iSpsSize,
	unsigned char*  pBufPps, int iBufPpsMax, int& iPpsSize)
{
	unsigned char aStartCode[4] = { 0, 0, 0, 1 };
	int ioffset = 0;
	unsigned char * pScan = pBuffer;
	unsigned char * pScanEnd = pBuffer + nSize - 4;
	unsigned char * pFindNalDataStart = NULL;
	unsigned char * pFindNalDataEnd = NULL;
	unsigned char   naluType = 0;

	iVpsSize = 0;
	iSpsSize = 0;
	iPpsSize = 0;

	while (pScan < pScanEnd)
	{
		if (iVpsSize !=0 && iSpsSize != 0 && iPpsSize != 0)
		{
			break;
		}

		if (memcmp(pScan, aStartCode + 1, 3) == 0)
		{
			if (pFindNalDataStart != NULL)
			{
				if ((pScan > pBuffer) && *(pScan - 1) == 0)
				{
					pFindNalDataEnd = pScan - 1;
				}
				else
				{
					pFindNalDataEnd = pScan;
				}
			}

			if (naluType == 32 && iVpsSize == 0)
			{
				if (iBufVpsMax > (4 + pFindNalDataEnd - pFindNalDataStart))
				{
					memcpy(pBufVps, aStartCode, 4);
					memcpy(pBufVps + 4, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
					iVpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
				}
				else
				{
					return 1;
				}
			}

			if (naluType == 33 && iSpsSize == 0)
			{
				if (iBufSpsMax > (4 + pFindNalDataEnd - pFindNalDataStart))
				{
					memcpy(pBufSps, aStartCode, 4);
					memcpy(pBufSps + 4, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
					iSpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
				}
				else
				{
					return 1;
				}
			}

			if (naluType == 34 && iPpsSize == 0)
			{
				if (iBufPpsMax > (4 + pFindNalDataEnd - pFindNalDataStart))
				{
					memcpy(pBufPps, aStartCode, 4);
					memcpy(pBufPps + 4, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
					iPpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
				}
				else
				{
					return 1;
				}
			}

			naluType = ((*(pScan + 3))>>1) & 0x03f;
			switch (naluType)
			{
				//VPS SPS PPS
				case 32:
				case 33:
				case 34:
				{
					pFindNalDataStart = pScan + 3;
					break;
				}

				default:
				{
					break;
				}
			}

			pScan += 3;
		}
		else
		{
			pScan++;
		}
	}
    
    if(pFindNalDataStart)
    {
        if (naluType == 32 && iVpsSize == 0)
        {
            memcpy(pBufVps, aStartCode, 4);
            memcpy(pBufVps + 4, pFindNalDataStart, nSize - (pFindNalDataStart - pBuffer));
            iVpsSize = 4 + nSize - (pFindNalDataStart - pBuffer);
        }
        else if (naluType == 33 && iSpsSize == 0)
        {
            memcpy(pBufSps, aStartCode, 4);
            memcpy(pBufSps + 4, pFindNalDataStart, nSize - (pFindNalDataStart - pBuffer));
            iSpsSize = 4 + nSize - (pFindNalDataStart - pBuffer);
        }
        else if (naluType == 34 && iPpsSize == 0)
        {
            memcpy(pBufPps, aStartCode, 4);
            memcpy(pBufPps + 4, pFindNalDataStart, nSize - (pFindNalDataStart - pBuffer));
            iPpsSize = 4 + nSize - (pFindNalDataStart - pBuffer);
        }
    }
    
	return 0;
}


int qcAV_ParseADTSAACHeaderInfo(unsigned char * pBuffer, int nSize, int *pOut_sampling_rate, int *pOut_channels, int *pOut_samplebitCount)
{
	unsigned char*   pFind = pBuffer;
	unsigned char*   pEnd = pBuffer + nSize - 5;
	unsigned int   ulFrameSize = 0;
	unsigned char     uProfile = 0;
	unsigned char     uSampleRateIndex = 0;
	unsigned char     uChannleCountIndex = 0;
	bool              bFindHeader = false;

	unsigned int aulSamplingRates[16] = { 96000, 88200, 64000, 48000, \
										  44100, 32000, 24000, 22050, \
										  16000, 12000, 11025, 8000, \
										  0, 0, 0, 0};

	unsigned char aChannels[8] = { 2, 1, 2, 3, 4, 5, 6, 7};

	while (pFind < pEnd && bFindHeader == false)
	{
		//ADTS Header 0xFF 0xFX
		if (pFind[0] == 0xFF && (pFind[1] & 0xF0) == 0xF0)
		{
			ulFrameSize = ((pFind[3] & 0x03) << 11) | (pFind[4] << 3) | (pFind[5] >> 5);

			//the buffer is one ADTS frame
			if (nSize == ulFrameSize )
			{
				//check Profile
				uProfile = pFind[2] >> 6;

				//check Sampling rate frequency index
				uSampleRateIndex = (pFind[2] >> 2) & 0xF;
				if (uSampleRateIndex > 0xB)
				{
					continue;
				}

				//check Channel configuration
				uChannleCountIndex = ((pFind[2] << 2) | (pFind[3] >> 6)) & 0x07;
				bFindHeader = true;
			}
			else
			{
				if ((int)(ulFrameSize) < nSize)
				{
					//the buffer is more than one ADTS frame
					if (*(pFind + ulFrameSize) == 0xFF && ((*(pFind + ulFrameSize + 1)) & 0xF0) == 0xF0)
					{
						//check Profile
						uProfile = pFind[2] >> 6;

						//check Sampling rate frequency index
						uSampleRateIndex = (pFind[2] >> 2) & 0xF;
						if (uSampleRateIndex > 0xB)
						{
							continue;
						}
						//check Channel configuration
						uChannleCountIndex = ((pFind[2] << 2) | (pFind[3] >> 6)) & 0x07;
						bFindHeader = true;
					}
				}
			}

			if (bFindHeader != true)
			{
				pFind++;
			}
		}
		else
		{
			pFind++;
		}
	}

	if (bFindHeader == true)
	{
		*pOut_sampling_rate = aulSamplingRates[uSampleRateIndex];
		*pOut_channels = aChannels[uChannleCountIndex];
		*pOut_samplebitCount = 0;
		return 0;
	}
	else
	{
		return 1;
	}
}

bool qcAV_IsHEVCReferenceFrame(unsigned char* pBuffer, int nSize)
{
    unsigned char nalHead[3] = {0, 0, 1};
    unsigned char* pScan = pBuffer;
    unsigned char* pScanEnd = pBuffer + nSize - 4;
    unsigned char  naluType = 0;
    while (pScan < pScanEnd)
    {
        if (memcmp(pScan, nalHead, 3) == 0)
        {
            naluType = ((*(pScan + 3))>>1) & 0x3f;
            if (naluType >= 19 && naluType <= 21)
            {
                return true;
            }
            else
            {
                pScan++;
            }
        }
        else
        {
            pScan++;
        }
    }
    
    return false;
}

int qcAV_IsNalUnit(unsigned char* pBuffer, int nSize)
{
    if (XRAW_IS_ANNEXB(pBuffer))
        return 3;
    if (XRAW_IS_ANNEXB2(pBuffer))
        return 4;
    
    return 0;
}

int qcAV_ConvertHEVCNalHead2 (unsigned char* pOutBuffer, int& nOutSize, unsigned char* pInBuffer, int nInSize, int &nNalLength)
{
    nNalLength = 4;
    nOutSize = 0;
    
    unsigned char aStartCode[4] = { 0, 0, 0, 1 };
    int ioffset = 0;
    unsigned char * pScan = pInBuffer;
    unsigned char * pScanEnd = pInBuffer + nInSize - 4;
    unsigned char * pFindNalDataStart = NULL;
    unsigned char * pFindNalDataEnd = NULL;
    unsigned char   naluType = 0;
    
    int nVpsSize = 0;
    int nSpsSize = 0;
    int nPpsSize = 0;
    
    while (pScan < pScanEnd)
    {
        if (nVpsSize !=0 && nSpsSize != 0 && nPpsSize != 0)
        {
            break;
        }
        
        if (memcmp(pScan, aStartCode + 1, 3) == 0)
        {
            if (pFindNalDataStart != NULL)
            {
                if ((pScan > pInBuffer) && *(pScan - 1) == 0)
                {
                    pFindNalDataEnd = pScan - 1;
                }
                else
                {
                    pFindNalDataEnd = pScan;
                }
            }
            
            if (naluType == 32 && nVpsSize == 0)
            {
                if (256 > (4 + pFindNalDataEnd - pFindNalDataStart))
                {
                    memcpy(pOutBuffer+nOutSize, aStartCode, 4);
                    nOutSize += 4;
                    memcpy(pOutBuffer+nOutSize, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
                    nOutSize += pFindNalDataEnd - pFindNalDataStart;
                    nVpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
                }
                else
                {
                    return 1;
                }
            }
            
            if (naluType == 33 && nSpsSize == 0)
            {
                if (256 > (4 + pFindNalDataEnd - pFindNalDataStart))
                {
                    memcpy(pOutBuffer+nOutSize, aStartCode, 4);
                    nOutSize += 4;
                    memcpy(pOutBuffer+nOutSize, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
                    nOutSize += pFindNalDataEnd - pFindNalDataStart;
                    nSpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
                }
                else
                {
                    return 1;
                }
            }
            
            if (naluType == 34 && nPpsSize == 0)
            {
                if (256 > (4 + pFindNalDataEnd - pFindNalDataStart))
                {
                    memcpy(pOutBuffer+nOutSize, aStartCode, 4);
                    nOutSize += 4;
                    memcpy(pOutBuffer+nOutSize, pFindNalDataStart, pFindNalDataEnd - pFindNalDataStart);
                    nOutSize += pFindNalDataEnd - pFindNalDataStart;
                    nPpsSize = 4 + pFindNalDataEnd - pFindNalDataStart;
                }
                else
                {
                    return 1;
                }
            }
            
            naluType = ((*(pScan + 3))>>1) & 0x03f;
            switch (naluType)
            {
                    //VPS SPS PPS
                case 32:
                case 33:
                case 34:
                {
                    pFindNalDataStart = pScan + 3;
                    break;
                }
                    
                default:
                {
                    break;
                }
            }
            
            pScan += 3;
        }
        else
        {
            pScan++;
        }
    }
    
    if(pFindNalDataStart)
    {
        if (naluType == 32 && nVpsSize == 0)
        {
            memcpy(pOutBuffer+nOutSize, aStartCode, 4);
            nOutSize += 4;
            memcpy(pOutBuffer+nOutSize, pFindNalDataStart, nInSize - (pFindNalDataStart - pInBuffer));
            nOutSize += (nInSize - (pFindNalDataStart - pInBuffer));
            nVpsSize = 4 + nInSize - (pFindNalDataStart - pInBuffer);
        }
        else if (naluType == 33 && nSpsSize == 0)
        {
            memcpy(pOutBuffer+nOutSize, aStartCode, 4);
            nOutSize += 4;
            memcpy(pOutBuffer+nOutSize, pFindNalDataStart, nInSize - (pFindNalDataStart - pInBuffer));
            nOutSize += (nInSize - (pFindNalDataStart - pInBuffer));
            nSpsSize = 4 + nInSize - (pFindNalDataStart - pInBuffer);
        }
        else if (naluType == 34 && nPpsSize == 0)
        {
            memcpy(pOutBuffer+nOutSize, aStartCode, 4);
            nOutSize += 4;
            memcpy(pOutBuffer+nOutSize, pFindNalDataStart, nInSize - (pFindNalDataStart - pInBuffer));
            nOutSize += (nInSize - (pFindNalDataStart - pInBuffer));
            nPpsSize = 4 + nInSize - (pFindNalDataStart - pInBuffer);
        }
    }
    
    return 0;
}
