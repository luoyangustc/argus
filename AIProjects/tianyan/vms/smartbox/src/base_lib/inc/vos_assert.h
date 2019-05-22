#ifndef __VOS_ASSERT_H__
#define __VOS_ASSERT_H__

#include "vos_types.h"

#if defined(VOS_HAS_ASSERT_H) && VOS_HAS_ASSERT_H != 0
#  include <assert.h>
#else
#  define assert(expr)
#endif


#ifndef vos_assert
#if 0
#   define vos_assert(expr)   assert(expr)
#else
	#   define vos_assert(expr)   \
		do { \
		if (!(expr)) {printf("%s:%d invoke assert!\n", __FILE__, __LINE__);}\
		assert(expr);\
		} while(0)
#endif
#endif

#if defined(VOS_ENABLE_EXTRA_CHECK) && VOS_ENABLE_EXTRA_CHECK != 0
#   define VOS_ASSERT_RETURN(expr,retval)    \
	    do { \
		if (!(expr)) { vos_assert(expr); return retval; } \
	    } while (0)
#else
#   define VOS_ASSERT_RETURN(expr,retval)    vos_assert(expr)
#endif

#if defined(VOS_ENABLE_EXTRA_CHECK) && VOS_ENABLE_EXTRA_CHECK != 0
#   define VOS_ASSERT_ON_FAIL(expr,exec_on_fail)    \
	    do { \
		vos_assert(expr); \
		if (!(expr)) exec_on_fail; \
	    } while (0)
#else
#   define VOS_ASSERT_ON_FAIL(expr,exec_on_fail)    vos_assert(expr)
#endif

#endif	/* __VOS_ASSERT_H__ */


