#define __VOS_FILE_IO_C__
#include <stdio.h>
#include "vos_socket.h"
#include "vos_assert.h"
#include "vos_file_io.h"

vos_status_t vos_file_open(const char *pathname, 
                                  unsigned flags,
                                  vos_oshandle_t *fd)
{
    char mode[8];
    char *p = mode;

    VOS_ASSERT_RETURN(pathname && fd, VOS_EINVAL);

    if ((flags & VOS_O_APPEND) == VOS_O_APPEND) 
	{
        if ((flags & VOS_O_WRONLY) == VOS_O_WRONLY) 
		{
            *p++ = 'a';
            *p++ = 'b';

            if ((flags & VOS_O_RDONLY) == VOS_O_RDONLY)
			{
                *p++ = '+';
			}
        }
		else 
		{
            /* This is invalid.
             * Can not specify VOS_O_RDONLY with VOS_O_APPEND! 
             */
        }
    } 
	else 
	{
        if ((flags & VOS_O_RDONLY) == VOS_O_RDONLY) 
		{
            *p++ = 'r';
            *p++ = 'b';
            if ((flags & VOS_O_WRONLY) == VOS_O_WRONLY)
			{
                *p++ = '+';
			}
        } 
		else 
		{
            *p++ = 'w';
            *p++ = 'b';
        }
    }

    if (p==mode)
	{
        return VOS_EINVAL;
	}
	
    *p++ = '\0';

    *fd = fopen(pathname, mode);
    if (*fd == NULL)
	{
        return VOS_RETURN_OS_ERROR(errno);
	}

    return VOS_SUCCESS;
}

vos_status_t vos_file_close(vos_oshandle_t fd)
{
    VOS_ASSERT_RETURN(fd, VOS_EINVAL);
    
	if (fclose((FILE*)fd) != 0)
	{
        return VOS_RETURN_OS_ERROR(errno);
	}

    return VOS_SUCCESS;
}

vos_status_t vos_file_write( vos_oshandle_t fd,
                                   const void *data,
                                   vos_ssize_t *size)
{
    size_t written;

    clearerr((FILE*)fd);
    written = fwrite(data, 1, *size, (FILE*)fd);
    if (ferror((FILE*)fd)) 
	{
        *size = -1;
        return VOS_RETURN_OS_ERROR(errno);
    }

    *size = written;
    return VOS_SUCCESS;
}

vos_status_t vos_file_read( vos_oshandle_t fd,
                                  void *data,
                                  vos_ssize_t *size)
{
    size_t bytes;

    clearerr((FILE*)fd);
    bytes = fread(data, 1, *size, (FILE*)fd);
    if (ferror((FILE*)fd)) 
	{
        *size = -1;
        return VOS_RETURN_OS_ERROR(errno);
    }

    *size = bytes;
    return VOS_SUCCESS;
}

/*
vos_bool_t vos_file_eof(vos_oshandle_t fd, enum vos_file_access access)
{
    VOS_UNUSED_ARG(access);
    return feof((FILE*)fd) ? TRUE : 0;
}
*/

vos_status_t vos_file_setpos( vos_oshandle_t fd,
                                    vos_offset_t offset,
                                    enum vos_file_seek_type whence)
{
    int mode;

    switch (whence) 
	{
    case VOS_SEEK_SET:
        mode = SEEK_SET; 
		break;
    case VOS_SEEK_CUR:
        mode = SEEK_CUR; 
		break;
    case VOS_SEEK_END:
        mode = SEEK_END; 
		break;
    default:
        vos_assert(!"Invalid whence in file_setpos");
        return VOS_EINVAL;
    }

    if (fseek((FILE*)fd, (long)offset, mode) != 0)
	{
        return VOS_RETURN_OS_ERROR(errno);
	}

    return VOS_SUCCESS;
}

vos_status_t vos_file_getpos( vos_oshandle_t fd,
                                    vos_offset_t *pos)
{
    long offset;

    offset = ftell((FILE*)fd);
    if (offset == -1) 
	{
        *pos = -1;
        return VOS_RETURN_OS_ERROR(errno);
    }

    *pos = offset;
    return VOS_SUCCESS;
}

vos_status_t vos_file_flush(vos_oshandle_t fd)
{
    int rc;

    rc = fflush((FILE*)fd);
    if (rc == EOF) 
	{
		return VOS_RETURN_OS_ERROR(errno);
    }

    return VOS_SUCCESS;
}

