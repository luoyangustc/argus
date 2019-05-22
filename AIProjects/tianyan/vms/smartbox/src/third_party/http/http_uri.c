
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "http_uri.h"
#include "vos_types.h"
#include "vos_os.h"
#include "vos_string.h"

#if (OS_WIN32 == 1)
#pragma warning(disable:4996)
#endif

typedef enum uri_parse_state_tag
{
  parse_state_read_host = 0,
  parse_state_read_port,
  parse_state_read_resource
} uri_parse_state;
    

int
http_uri_parse(char *a_string,
	       http_uri *a_uri)
{
  /* Everyone chant... "we love state machines..." */
  uri_parse_state l_state = parse_state_read_host;
  char *l_start_string = NULL;
  char *l_end_string = NULL;
  char  l_temp_port[6];

  /* init the array */
  memset(l_temp_port, 0, 6);
  /* check the parameters */
  if (a_string == NULL)
    goto ec;
  if (a_uri) {
    a_uri->full = vos_strdup(a_string);
  }
  l_start_string = strchr(a_string, ':');
  /* check to make sure that there was a : in the string */
  if (!l_start_string)
    goto ec;
  if (a_uri) {
    a_uri->proto = VOS_MALLOC_BLK_T( char, (l_start_string - a_string + 1) );//(char *)malloc(l_start_string - a_string + 1);
    memcpy(a_uri->proto, a_string, (l_start_string - a_string));
    a_uri->proto[l_start_string - a_string] = '\0';
  }
  /* check to make sure it starts with "http://" */
  if (strncmp(l_start_string, "://", 3) != 0)
    goto ec;
  /* start at the beginning of the string */
  l_start_string = l_end_string = &l_start_string[3];
  while(*l_end_string)
    {
      if (l_state == parse_state_read_host)
	{
	  if (*l_end_string == ':')
	    {
	      l_state = parse_state_read_port;
	      if ((l_end_string - l_start_string) == 0)
		goto ec;
	      /* allocate space */
	      if ((l_end_string - l_start_string) == 0)
		goto ec;
	      /* only do this if a uri was passed in */
	      if (a_uri)
		{
		  a_uri->host = VOS_MALLOC_BLK_T( char, (l_end_string - l_start_string + 1) );//(char *)malloc(l_end_string - l_start_string + 1);
		  /* copy the data */
		  memcpy(a_uri->host, l_start_string, (l_end_string - l_start_string));
		  /* terminate */
		  a_uri->host[l_end_string - l_start_string] = '\0';
		}
	      /* reset the counters */
	      l_end_string++;
	      l_start_string = l_end_string;
	      continue;
	    }
	  else if (*l_end_string == '/')
	    {
	      l_state = parse_state_read_resource;
	      if ((l_end_string - l_start_string) == 0)
		goto ec;
	      if (a_uri)
		{
		  a_uri->host = VOS_MALLOC_BLK_T( char, (l_end_string - l_start_string + 1) );//(char *)malloc(l_end_string - l_start_string + 1);
		  memcpy(a_uri->host, l_start_string, (l_end_string - l_start_string));
		  a_uri->host[l_end_string - l_start_string] = '\0';
		}
	      l_start_string = l_end_string;
	      continue;
	    }
	}
      else if (l_state == parse_state_read_port)
	{
	  if (*l_end_string == '/')
	    {
	      l_state = parse_state_read_resource;
	      /* check to make sure we're not going to overflow */
	      if (l_end_string - l_start_string > 5)
		goto ec;
	      /* check to make sure there was a port */
	      if ((l_end_string - l_start_string) == 0)
		goto ec;
	      /* copy the port into a temp buffer */
	      memcpy(l_temp_port, l_start_string, l_end_string - l_start_string);
	      /* convert it. */
	      if (a_uri)
		a_uri->port = atoi(l_temp_port);
	      l_start_string = l_end_string;
	      continue;
	    }
	  else if (isdigit(*l_end_string) == 0)
	    {
	      /* check to make sure they are just digits */
	      goto ec;
	    }
	}
      /* next.. */
      l_end_string++;
      continue;
    }
  
  if (l_state == parse_state_read_host)
    {
      if ((l_end_string - l_start_string) == 0)
	goto ec;
      if (a_uri)
	{
	  a_uri->host = VOS_MALLOC_BLK_T( char, (l_end_string - l_start_string + 1) );
	  memcpy(a_uri->host, l_start_string, (l_end_string - l_start_string));
	  a_uri->host[l_end_string - l_start_string] = '\0';
	  /* for a "/" */
	  a_uri->resource = vos_strdup("/");
	}
    }
  else if (l_state == parse_state_read_port)
    {
      if (strlen(l_start_string) == 0)
	/* oops.  that's not a valid number */
	goto ec;
      if (a_uri)
	{
	  a_uri->port = atoi(l_start_string);
	  a_uri->resource = vos_strdup("/");
	}
    }
  else if (l_state == parse_state_read_resource)
    {
      if (strlen(l_start_string) == 0)
	{
	  if (a_uri)
	    a_uri->resource = vos_strdup("/");
	}
      else
	{
	  if (a_uri)
	    a_uri->resource = vos_strdup(l_start_string);
	}
    }
  else
    {
      /* uhh...how did we get here? */
      goto ec;
    }
  return 0;
	  
 ec:
  return -1;
}

http_uri *
http_uri_new(void)
{
  http_uri *l_return = NULL;

  
  l_return = VOS_MALLOC_T(http_uri);
  l_return->full = NULL;
  l_return->proto = NULL;
  l_return->host = NULL;
  l_return->port = 80;
  l_return->resource = NULL;
  return l_return;
}

void
http_uri_destroy(http_uri *a_uri)
{
  if (a_uri->full) {
    //VOS_PRINTF("http_uri_destroy--->free a_uri->full\n");
    VOS_FREE_T(a_uri->full);
    a_uri->full = NULL;
  }
  if (a_uri->proto) {
    //VOS_PRINTF("http_uri_destroy--->free a_uri->proto\n");
    VOS_FREE_T(a_uri->proto);
    a_uri->proto = NULL;
  }
  if (a_uri->host) {
    //VOS_PRINTF("http_uri_destroy--->free a_uri->host\n");
    VOS_FREE_T(a_uri->host);
    a_uri->host = NULL;
  }
  if (a_uri->resource) {
    //VOS_PRINTF("http_uri_destroy--->free a_uri->resource\n");
    VOS_FREE_T(a_uri->resource);
    a_uri->resource = NULL;
  }
  //VOS_PRINTF("http_uri_destroy--->free a_uri");
  VOS_FREE_T(a_uri);
}


