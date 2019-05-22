
#ifndef __VOS_FILE_IO_H__
#define __VOS_FILE_IO_H__


#include "vos_types.h"

#undef  EXT
#ifdef __VOS_FILE_IO_C__
#define EXT 
#else
#define EXT extern
#endif

VOS_BEGIN_DECL 

/**
 * @defgroup VOS_FILE_IO File I/O
 * @ingroup VOS_IO
 * @{
 *
 * This file contains functionalities to perform file I/O. The file
 * I/O can be implemented with various back-end, either using native
 * file API or ANSI stream. 
 *
 * @section vos_file_size_limit_sec Size Limits
 *
 * There may be limitation on the size that can be handled by the
 * #vos_file_setpos() or #vos_file_getpos() functions. The API itself
 * uses 64-bit integer for the file offset/position (where available); 
 * however some backends (such as ANSI) may only support signed 32-bit 
 * offset resolution.
 *
 * Reading and writing operation uses signed 32-bit integer to indicate
 * the size.
 *
 *
 */

/**
 * These enumerations are used when opening file. Values VOS_O_RDONLY,
 * VOS_O_WRONLY, and VOS_O_RDWR are mutually exclusive. Value VOS_O_APPEND
 * can only be used when the file is opened for writing. 
 */
enum vos_file_access
{
    VOS_O_RDONLY     = 0x1101,   /**< Open file for reading.             */
    VOS_O_WRONLY     = 0x1102,   /**< Open file for writing.             */
    VOS_O_RDWR       = 0x1103,   /**< Open file for reading and writing. 
                                     File will be truncated.            */
    VOS_O_APPEND     = 0x1108    /**< Append to existing file.           */
};

/**
 * The seek directive when setting the file position with #vos_file_setpos.
 */
enum vos_file_seek_type
{
    VOS_SEEK_SET     = 0x1201,   /**< Offset from beginning of the file. */
    VOS_SEEK_CUR     = 0x1202,   /**< Offset from current position.      */
    VOS_SEEK_END     = 0x1203    /**< Size of the file plus offset.      */
};

/**
 * Open the file as specified in \c pathname with the specified
 * mode, and return the handle in \c fd. All files will be opened
 * as binary.
 *
 * @param pathname      The file name to open.
 * @param flags         Open flags, which is bitmask combination of
 *                      #vos_file_access enum. The flag must be either
 *                      VOS_O_RDONLY, VOS_O_WRONLY, or VOS_O_RDWR. When file
 *                      writing is specified, existing file will be 
 *                      truncated unless VOS_O_APPEND is specified.
 * @param fd            The returned descriptor.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_open(const char *pathname, 
                                  unsigned flags,
                                  vos_oshandle_t *fd);

/**
 * Close an opened file descriptor.
 *
 * @param fd            The file descriptor.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_close(vos_oshandle_t fd);

/**
 * Write data with the specified size to an opened file.
 *
 * @param fd            The file descriptor.
 * @param data          Data to be written to the file.
 * @param size          On input, specifies the size of data to be written.
 *                      On return, it contains the number of data actually
 *                      written to the file.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_write(vos_oshandle_t fd,
                                   const void *data,
                                   vos_ssize_t *size);

/**
 * Read data from the specified file. When end-of-file condition is set,
 * this function will return VOS_SUCCESS but the size will contain zero.
 *
 * @param fd            The file descriptor.
 * @param data          Pointer to buffer to receive the data.
 * @param size          On input, specifies the maximum number of data to
 *                      read from the file. On output, it contains the size
 *                      of data actually read from the file. It will contain
 *                      zero when EOF occurs.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 *                      When EOF occurs, the return is VOS_SUCCESS but size
 *                      will report zero.
 */
vos_status_t vos_file_read(vos_oshandle_t fd,
                                  void *data,
                                  vos_ssize_t *size);

/**
 * Set file position to new offset according to directive \c whence.
 *
 * @param fd            The file descriptor.
 * @param offset        The new file position to set.
 * @param whence        The directive.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_setpos(vos_oshandle_t fd,
                                    vos_offset_t offset,
                                    enum vos_file_seek_type whence);

/**
 * Get current file position.
 *
 * @param fd            The file descriptor.
 * @param pos           On return contains the file position as measured
 *                      from the beginning of the file.
 *
 * @return              VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_getpos(vos_oshandle_t fd,
                                    vos_offset_t *pos);

/**
 * Flush file buffers.
 *
 * @param fd		The file descriptor.
 *
 * @return		VOS_SUCCESS or the appropriate error code on error.
 */
vos_status_t vos_file_flush(vos_oshandle_t fd);

VOS_END_DECL

#endif  /* __VOS_FILE_IO_H__ */


