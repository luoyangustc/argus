
#ifndef HTTP_URI_H
#define HTTP_URI_H

/* strings that are used all over the place */


typedef struct http_uri_tag
{
  char             *full;                          /* full URL */
  char             *proto;                         /* protocol */
  char             *host;                          /* copy semantics */
  unsigned short    port;
  char             *resource;                      /* copy semantics */
} http_uri;

http_uri *
http_uri_new(void);
   
void
http_uri_destroy(http_uri *a_uri);

int
http_uri_parse(char *a_uri,
	       http_uri *a_request);

#endif



