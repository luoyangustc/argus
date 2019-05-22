
#ifndef HTTP_HDRS_H
#define HTTP_HDRS_H

#include "ghttp_constants.h"

#include "vos_types.h"


/* the list of known headers */
extern const char *http_hdr_known_list[];

/* a header list */
#define HTTP_HDRS_MAX 256
typedef struct http_hdr_list_tag
{
  char *header[HTTP_HDRS_MAX];
  char *value[HTTP_HDRS_MAX];
} http_hdr_list;

/* functions dealing with headers */

/* check to see if the library knows about the header */
const char *
http_hdr_is_known(const char *a_hdr);

/* create a new list */
http_hdr_list *
http_hdr_list_new(void);

/* destroy a list */
void
http_hdr_list_destroy(http_hdr_list *a_list);

/* set a value in a list */
int
http_hdr_set_value(http_hdr_list *a_list,
		   const char *a_name,
		   const char *a_val);

/* set the value in a list from a range, not a NTS */
int
http_hdr_set_value_no_nts(http_hdr_list *a_list,
			  const char *a_name_start,
			  int a_name_len,
			  const char *a_val_start,
			  int a_val_len);

/* get a copy of a value in a list */
char *
http_hdr_get_value(http_hdr_list *a_list,
		   const char *a_name);

/* get a copy of the headers in a list */
int
http_hdr_get_headers(http_hdr_list *a_list,
		     char ***a_names,
		     int *a_num_names);

/* clear a header in a list */
int
http_hdr_clear_value(http_hdr_list *a_list,
		     const char *a_name);


#endif /* HTTP_HDRS_H */
