
#ifndef GHTTP_CONSTANTS_H
#define GHTTP_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern const char http_hdr_Allow[];
extern const char http_hdr_Content_Encoding[];
extern const char http_hdr_Content_Language[];
extern const char http_hdr_Content_Length[];
extern const char http_hdr_Content_Location[];
extern const char http_hdr_Content_MD5[];
extern const char http_hdr_Content_Range[];
extern const char http_hdr_Content_Type[];
extern const char http_hdr_Expires[];
extern const char http_hdr_Last_Modified[];

/* general headers */

extern const char http_hdr_Cache_Control[];
extern const char http_hdr_Connection[];
extern const char http_hdr_Date[];
extern const char http_hdr_Pragma[];
extern const char http_hdr_Transfer_Encoding[];
extern const char http_hdr_Update[];
extern const char http_hdr_Trailer[];
extern const char http_hdr_Via[];

/* request headers */

extern const char http_hdr_Accept[];
extern const char http_hdr_Accept_Charset[];
extern const char http_hdr_Accept_Encoding[];
extern const char http_hdr_Accept_Language[];
extern const char http_hdr_Authorization[];
extern const char http_hdr_Expect[];
extern const char http_hdr_From[];
extern const char http_hdr_Host[];
extern const char http_hdr_If_Modified_Since[];
extern const char http_hdr_If_Match[];
extern const char http_hdr_If_None_Match[];
extern const char http_hdr_If_Range[];
extern const char http_hdr_If_Unmodified_Since[];
extern const char http_hdr_Max_Forwards[];
extern const char http_hdr_Proxy_Authorization[];
extern const char http_hdr_Range[];
extern const char http_hdr_Referrer[];
extern const char http_hdr_TE[];
extern const char http_hdr_User_Agent[];

/* response headers */

extern const char http_hdr_Accept_Ranges[];
extern const char http_hdr_Age[];
extern const char http_hdr_ETag[];
extern const char http_hdr_Location[];
extern const char http_hdr_Retry_After[];
extern const char http_hdr_Server[];
extern const char http_hdr_Vary[];
extern const char http_hdr_Warning[];
extern const char http_hdr_WWW_Authenticate[];

/* Other headers */

extern const char http_hdr_Set_Cookie[];

/* WebDAV headers */

extern const char http_hdr_DAV[];
extern const char http_hdr_Depth[];
extern const char http_hdr_Destination[];
extern const char http_hdr_If[];
extern const char http_hdr_Lock_Token[];
extern const char http_hdr_Overwrite[];
extern const char http_hdr_Status_URI[];
extern const char http_hdr_Timeout[];

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* GHTTP_CONSTANTS_H */
