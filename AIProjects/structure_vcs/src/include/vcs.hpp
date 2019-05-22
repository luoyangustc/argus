#ifndef __VCS_H__
#define __VCS_H__

// #include <stdio.h>

extern "C"
{

typedef enum
{	
	status_Success		   = 1,
	status_ErrInputParam,
	status_Error		   = 255,
	status_UnknowError,
	
}statusCode;

typedef void (*vid_analysis_cb_func)(const char* json_str, const char* channel_id);

statusCode initAnalysisModule(const char* json_str, int json_str_len);
statusCode uninitAnalysisModule();
statusCode startVidAnalysis(const char* url, int url_len, const char* channel_id, int channel_id_len, vid_analysis_cb_func algo_callback);
statusCode stopVidAnalysis(const char* channel_id, int channel_id_len);
const char* getVidChannelAnalysisStatus();

}
#endif
