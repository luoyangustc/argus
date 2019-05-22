
#ifndef __VOS_FILE_ACCESS_H__
#define __VOS_FILE_ACCESS_H__


#include "vos_types.h"

#undef  EXT
#ifdef __VOS_FILE_ACCESS_C__
#define EXT 
#else
#define EXT extern
#endif

VOS_BEGIN_DECL 

/**
 * This structure describes file information, to be obtained by
 * calling #vos_file_getstat(). The time information in this structure
 * is in local time.
 */
typedef struct vos_file_stat
{
    vos_offset_t        size;   /**< Total file size.               */
    vos_time_val     atime;  /**< Time of last access.           */
    vos_time_val     mtime;  /**< Time of last modification.     */
    vos_time_val     ctime;  /**< Time of last creation.         */
} vos_file_stat;


/**
 * Returns non-zero if the specified file exists.
 *
 * @param filename      The file name.
 *
 * @return              Non-zero if the file exists.
 */
EXT vos_bool_t vos_file_exists(const char *filename, vos_bool_t dir);

/**
 * Returns the size of the file.
 *
 * @param filename      The file name.
 *
 * @return              The file size in bytes or -1 on error.
 */
EXT vos_offset_t vos_file_size(const char *filename);

/**
 * Delete a file.
 *
 * @param filename      The filename.
 *
 * @return              VOS_SUCCESS on success or the appropriate error code.
 */
EXT vos_status_t vos_file_delete(const char *filename);

/**
 * Move a \c oldname to \c newname. If \c newname already exists,
 * it will be overwritten.
 *
 * @param oldname       The file to rename.
 * @param newname       New filename to assign.
 *
 * @return              VOS_SUCCESS on success or the appropriate error code.
 */
EXT vos_status_t vos_file_move( const char *oldname, 
                                   const char *newname);


/**
 * Return information about the specified file. The time information in
 * the \c stat structure will be in local time.
 *
 * @param filename      The filename.
 * @param stat          Pointer to variable to receive file information.
 *
 * @return              VOS_SUCCESS on success or the appropriate error code.
 */
EXT vos_status_t vos_file_getstat(const char *filename, vos_file_stat *statbuf);

EXT vos_status_t vos_file_mkdir(const char *dirname);

VOS_END_DECL


#endif	/* __VOS_FILE_ACCESS_H__ */

