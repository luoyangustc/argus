#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__


#define SAFE_FREE(a)					  \
        if((a) != NULL)                   \
																				        {                                 \
			free((a));					  \
			(a) = NULL;                   \
																						};                                \


#define BREAK_IN_ERR_NOT_ZERO(a, b, c)    \
        if((a) != 0)                      \
																				        {                                 \
			(b) = (c);					  \
			break;						  \
																						};                                \


#define BREAK_IN_ERR_ZERO(a, b, c)        \
        if((a) == 0)                      \
																				        {                                 \
			(b) = (c); \
			break;						  \
																						};                                \


#define BREAK_IN_NULL_POINTER(a, b, c)    \
        if((a) == NULL)                   \
																				        {                                 \
			(b) = (c);                    \
			break;						  \
																						};                                \

#define ERR_CURL_INNNER  2
#define ERR_LACK_MEMORY  3
#define ERR_INVALID_PARAMETER  4
#define ERR_NOT_ENOUGH_BUF     5
#define ERR_OPEN_URL_FAIL      6
#define ERR_ALREADY_EXSIT      7


#endif