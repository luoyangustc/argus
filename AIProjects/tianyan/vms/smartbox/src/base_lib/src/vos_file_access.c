#define  __VOS_FILE_ACCESS_C__

#include "vos_file_access.h"
#include "vos_assert.h"
#include "vos_os.h"
#include "vos_string.h"
#include "vos_time.h"

#if (OS_WIN32 == 1)
	#include <windows.h>
	#include "vos_unicode.h"
	#define CONTROL_ACCESS   READ_CONTROL
#else
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <unistd.h>
	#include <stdio.h>	/* rename() */
	#include <errno.h>
	//#include <dirent.h>	/*opendir()*/
#endif

/*
 * vos_file_exists()
 */
vos_bool_t vos_file_exists(const char *filename, vos_bool_t dir)
{
#if (OS_WIN32 == 1)
	VOS_DECL_UNICODE_TEMP_BUF(wfilename,256)
	VOS_DECL_UNICODE_TEMP_BUF(wfilepre,8)
	HANDLE hFile;

	VOS_ASSERT_RETURN(filename != NULL, 0);

	if(dir)
	{
		hFile = CreateFile(VOS_STRING_TO_NATIVE(filename, wfilename, sizeof(wfilename)), 
			FILE_LIST_DIRECTORY, 
			FILE_SHARE_READ, NULL,
			OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
	}
	else
	{
		hFile = CreateFile(VOS_STRING_TO_NATIVE(filename,wfilename,sizeof(wfilename)), 
			CONTROL_ACCESS, 
			FILE_SHARE_READ, NULL,
			OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	}
	if (hFile == INVALID_HANDLE_VALUE)
	{
		return FALSE;
	}
	CloseHandle(hFile);
	return TRUE;
#else
    struct stat buf;

    VOS_ASSERT_RETURN(filename, 0);

    if (stat(filename, &buf) != 0)
	{
		return FALSE;
	}

	if(dir && !S_ISDIR(buf.st_mode))
	{
		return FALSE;
	}
	return TRUE;
#endif
}

/*
 * vos_file_size()
 */
vos_offset_t vos_file_size(const char *filename)
{
#if (OS_WIN32 == 1)
	VOS_DECL_UNICODE_TEMP_BUF(wfilename,256)
	HANDLE hFile;
	DWORD sizeLo, sizeHi;
	vos_offset_t size;

	VOS_ASSERT_RETURN(filename != NULL, -1);

	hFile = CreateFile(VOS_STRING_TO_NATIVE(filename, wfilename,sizeof(wfilename)), 
						CONTROL_ACCESS, 
						FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
						OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		return -1;
	}

	sizeLo = GetFileSize(hFile, &sizeHi);
	if (sizeLo == INVALID_FILE_SIZE) 
	{
		DWORD dwStatus = GetLastError();
		if (dwStatus != NO_ERROR) 
		{
			CloseHandle(hFile);
			return -1;
		}
	}

	size = sizeHi;
	size = (size << 32) + sizeLo;

	CloseHandle(hFile);
	return size;
#else
	struct stat buf;

    VOS_ASSERT_RETURN(filename, -1);

    if (stat(filename, &buf) != 0)
	{
		return -1;
	}

    return buf.st_size;
#endif
}


/*
 * vos_file_delete()
 */
vos_status_t vos_file_delete(const char *filename)
{
#if (OS_WIN32 == 1)
	DWORD error;
	VOS_DECL_UNICODE_TEMP_BUF(wfilename,256)

	VOS_ASSERT_RETURN(filename != NULL, VOS_EINVAL);

    if (DeleteFile(VOS_STRING_TO_NATIVE(filename,wfilename,sizeof(wfilename))) == FALSE)
    {
        error = GetLastError();
        return VOS_RETURN_OS_ERROR(error);
    }
	return VOS_SUCCESS;
#else
	VOS_ASSERT_RETURN(filename, VOS_EINVAL);

    if (unlink(filename)!=0) 
	{
		return VOS_RETURN_OS_ERROR(errno);
    }
    return VOS_SUCCESS;
#endif
}

#if (OS_WIN32 == 1)
static vos_status_t file_time_to_time_val(const FILETIME *file_time,
	vos_time_val *time_val)
{
	FILETIME local_file_time;
	SYSTEMTIME localTime;
	vos_parsed_time pt;

	if (!FileTimeToLocalFileTime(file_time, &local_file_time))
		return VOS_RETURN_OS_ERROR(GetLastError());

	if (!FileTimeToSystemTime(file_time, &localTime))
		return VOS_RETURN_OS_ERROR(GetLastError());

	//if (!SystemTimeToTzSpecificLocalTime(NULL, &systemTime, &localTime))
	//    return VOS_RETURN_OS_ERROR(GetLastError());

	vos_bzero(&pt, sizeof(pt));
	pt.year = localTime.wYear;
	pt.mon = localTime.wMonth-1;
	pt.day = localTime.wDay;
	pt.wday = localTime.wDayOfWeek;

	pt.hour = localTime.wHour;
	pt.min = localTime.wMinute;
	pt.sec = localTime.wSecond;
	pt.msec = localTime.wMilliseconds;

	return vos_time_encode(&pt, time_val);
}
#endif

/*
 * vos_file_move()
 */
vos_status_t vos_file_move( const char *oldname, const char *newname)
{
#if (OS_WIN32 == 1)
	VOS_DECL_UNICODE_TEMP_BUF(woldname,256)
	VOS_DECL_UNICODE_TEMP_BUF(wnewname,256)
	BOOL rc;

	VOS_ASSERT_RETURN(oldname!=NULL && newname!=NULL, VOS_EINVAL);

	#if VOS_WIN32_WINNT >= 0x0400
	rc = MoveFileEx(VOS_STRING_TO_NATIVE(oldname,woldname,sizeof(woldname)), 
					VOS_STRING_TO_NATIVE(newname,wnewname,sizeof(wnewname)), 
					MOVEFILE_COPY_ALLOWED|MOVEFILE_REPLACE_EXISTING);
	#else
	rc = MoveFile(VOS_STRING_TO_NATIVE(oldname,woldname,sizeof(woldname)), 
					VOS_STRING_TO_NATIVE(newname,wnewname,sizeof(wnewname)));
	#endif
	if (!rc)
	{
		return VOS_RETURN_OS_ERROR(GetLastError());
	}
	return VOS_SUCCESS;
#else
    VOS_ASSERT_RETURN(oldname && newname, VOS_EINVAL);
    if (rename(oldname, newname) != 0) 
	{
		return VOS_RETURN_OS_ERROR(errno);
    }
    return VOS_SUCCESS;
#endif
}


/*
 * vos_file_getstat()
 */
vos_status_t vos_file_getstat(const char *filename, 
				    vos_file_stat *statbuf)
{
#if (OS_WIN32 == 1)
	VOS_DECL_UNICODE_TEMP_BUF(wfilename,256)
	HANDLE hFile;
	DWORD sizeLo, sizeHi;
	FILETIME creationTime, accessTime, writeTime;

	VOS_ASSERT_RETURN(filename!=NULL && statbuf!=NULL, VOS_EINVAL);

	hFile = CreateFile(VOS_STRING_TO_NATIVE(filename,wfilename,sizeof(wfilename)), 
						CONTROL_ACCESS, 
						FILE_SHARE_READ, NULL,
						OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
	{
		return VOS_RETURN_OS_ERROR(GetLastError());
	}
	sizeLo = GetFileSize(hFile, &sizeHi);
	if (sizeLo == INVALID_FILE_SIZE) 
	{
		DWORD dwStatus = GetLastError();
		if (dwStatus != NO_ERROR) 
		{
			CloseHandle(hFile);
			return VOS_RETURN_OS_ERROR(dwStatus);
		}
	}

	statbuf->size = sizeHi;
	statbuf->size = (statbuf->size << 32) + sizeLo;

	if (GetFileTime(hFile, &creationTime, &accessTime, &writeTime)==FALSE) 
	{
		DWORD dwStatus = GetLastError();
		CloseHandle(hFile);
		return VOS_RETURN_OS_ERROR(dwStatus);
	}

	CloseHandle(hFile);

	if (file_time_to_time_val(&creationTime, &statbuf->ctime) != VOS_SUCCESS)
	{
		return VOS_RETURN_OS_ERROR(GetLastError());
	}

	file_time_to_time_val(&accessTime, &statbuf->atime);
	file_time_to_time_val(&writeTime, &statbuf->mtime);

	return VOS_SUCCESS;
#else
    struct stat buf;
    VOS_ASSERT_RETURN(filename && statbuf, VOS_EINVAL);
    if (stat(filename, &buf) != 0) 
	{
		return VOS_RETURN_OS_ERROR(errno);
    }
    statbuf->size = buf.st_size;
    statbuf->ctime.sec = buf.st_ctime;
    statbuf->ctime.usec = 0;
    statbuf->mtime.sec = buf.st_mtime;
    statbuf->mtime.usec = 0;
    statbuf->atime.sec = buf.st_atime;
    statbuf->atime.usec = 0;
    return VOS_SUCCESS;
#endif
}

vos_status_t vos_file_mkdir(const char *dirname)
{
#if (OS_WIN32 == 1)
	return CreateDirectory(dirname, NULL)?VOS_SUCCESS:-1;
#else
	return mkdir(dirname, S_IRWXU);
#endif
}

