
#ifndef HTTP_RESP_H
#define HTTP_RESP_H

#include "http_hdrs.h"
#include "http_trans.h"
#include "http_req.h"

#include "vos_types.h"

#define HTTP_RESP_INFORMATIONAL(x) (x >=100 && < 200)
#define HTTP_RESP_SUCCESS(x) (x >= 200 && x < 300)
#define HTTP_RESP_REDIR(x) (x >= 300 && x < 400)
#define HTTP_RESP_CLIENT_ERR(x) (x >= 400 && x < 500)
#define HTTP_RESP_SERVER_ERR(x) (x >= 500 && x < 600)

typedef enum http_resp_header_state_tag
{
  http_resp_header_start = 0,
  http_resp_reading_header
} http_resp_header_state;

typedef enum http_resp_body_state_tag
{
  http_resp_body_start = 0,
  http_resp_body_read_content_length,
  http_resp_body_read_chunked,
  http_resp_body_read_standard
} http_resp_body_state;



typedef struct http_resp_tag
{
  float                                http_ver;
  int                                  status_code;
  char                                *reason_phrase;
  http_hdr_list                       *headers;
  char                                *body;
  int                                  body_len;
  int                                  content_length;
  int                                  flushed_length;
  http_resp_header_state               header_state;
  http_resp_body_state                 body_state;
} http_resp;

http_resp *
http_resp_new(void);

void
http_resp_destroy(http_resp *a_resp);

int
http_resp_read_body(http_resp *a_resp,
		    http_req *a_req,
		    http_trans_conn *a_conn);

int
http_resp_read_headers(http_resp *a_resp, http_trans_conn *a_conn);

void
http_resp_flush(http_resp *a_resp,
                http_trans_conn *a_conn);

#endif /* HTTP_RESP_H */
