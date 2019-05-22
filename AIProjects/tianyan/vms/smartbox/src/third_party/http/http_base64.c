
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "vos_types.h"

static const char b64_alphabet[65] = { 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/=" };
    
char *
http_base64_encode(const char *text) {
  /* The tricky thing about this is doing the padding at the end,
   * doing the bit manipulation requires a bit of concentration only */
  char *buffer = NULL;
  char *point = NULL;
  int inlen = 0;
  int outlen = 0;

  /* check our args */
  if (text == NULL)
    return NULL;
  
  /* Use 'buffer' to store the output. Work out how big it should be...
   * This must be a multiple of 4 bytes */
  
  inlen = strlen( text );
  /* check our arg...avoid a pesky FPE */
  if (inlen == 0)
    {
      buffer = VOS_MALLOC_T(char);//malloc(sizeof(char));
      buffer[0] = '\0';
      return buffer;
    }
  outlen = (inlen*4)/3;
  if( (inlen % 3) > 0 ) /* got to pad */
    outlen += 4 - (inlen % 3);
  
  buffer = VOS_MALLOC_BLK_T(char, outlen + 1);//malloc( outlen + 1 ); /* +1 for the \0 */
  memset(buffer, 0, outlen + 1); /* initialize to zero */
  
  /* now do the main stage of conversion, 3 bytes at a time,
   * leave the trailing bytes (if there are any) for later */
  
  for( point=buffer; inlen>=3; inlen-=3, text+=3 ) {
    *(point++) = b64_alphabet[ *text>>2 ]; 
    *(point++) = b64_alphabet[ (*text<<4 & 0x30) | *(text+1)>>4 ]; 
    *(point++) = b64_alphabet[ (*(text+1)<<2 & 0x3c) | *(text+2)>>6 ];
    *(point++) = b64_alphabet[ *(text+2) & 0x3f ];
  }
  
  /* Now deal with the trailing bytes */
  if( inlen ) {
    /* We always have one trailing byte */
    *(point++) = b64_alphabet[ *text>>2 ];
    *(point++) = b64_alphabet[ (*text<<4 & 0x30) |
			     (inlen==2?*(text+1)>>4:0) ]; 
    *(point++) = (inlen==1?'=':b64_alphabet[ *(text+1)<<2 & 0x3c ] );
    *(point++) = '=';
  }
  
  *point = '\0';
  
  return buffer;
}

